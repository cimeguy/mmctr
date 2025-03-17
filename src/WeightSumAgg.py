

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSumAggregation(nn.Module):
    def __init__(self, embedding_dim):
        """
        基于加权求和的局部无序性建模
        :param embedding_dim: 嵌入的维度
        """
        super(WeightedSumAggregation, self).__init__()
        # 注意力网络，用于生成权重分数
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)  # 输出一个标量分数
        )

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, seq_length, num_ids, embedding_dim)
        :return: 聚合后的张量，形状为 (batch_size, seq_length, embedding_dim)
        """
        batch_size, seq_length, num_ids, embedding_dim = x.size()

        # Step 1: 计算注意力分数
        attn_scores = self.attention(x)  # (batch_size, seq_length, num_ids, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch_size, seq_length, num_ids)

        # Step 2: 对注意力分数进行归一化（Softmax）
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_length, num_ids)

        # Step 3: 加权求和
        weighted_sum = torch.einsum('bsnd,bsn->bsd', x, attn_weights)  # (batch_size, seq_length, embedding_dim)

        return weighted_sum