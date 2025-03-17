
# =========================================================================
#aborted
# =========================================================================

import torch
from torch import nn
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, DIN_Attention, Dice
from fuxictr.utils import not_in_whitelist


import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, emb_size=256, num_heads=4, num_layers=2, ff_hidden_dim=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.emb_size = emb_size
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_size, emb_size)

    def forward(self, sequence_emb, target_emb, mask):
        # Transformer expects input of shape [seq_len, batchsize, emb_size]
        sequence_emb = sequence_emb.permute(1, 0, 2)  # [seq_len, batchsize, emb_size]
        
        # Create the key_padding_mask
        key_padding_mask = mask == 0  # Convert 0/1 mask to True/False
        
        # Pass through Transformer Encoder
        encoded_seq = self.transformer_encoder(sequence_emb, src_key_padding_mask=key_padding_mask)
        
        # Back to original shape
        encoded_seq = encoded_seq.permute(1, 0, 2)  # [batchsize, seq_len, emb_size]
        
        # 聚合 seq_len 维度
        # 方法 1: 平均池化
        encoded_seq = encoded_seq.mean(dim=1)  # [batchsize, emb_size]
        
        # 方法 2: 最大池化
        # encoded_seq, _ = encoded_seq.max(dim=1)  # [batchsize, emb_size]
        
        # 方法 3: 最后一个时间步
        # encoded_seq = encoded_seq[:, -1, :]  # [batchsize, emb_size]
        
        # 方法 4: 结合全连接层
        # encoded_seq = self.fc(encoded_seq.mean(dim=1))  # [batchsize, emb_size]
        
        return encoded_seq


class Transformer_ID_WithCLS(nn.Module):
    def __init__(self, num_ids, embedding_dim, num_heads, num_layers, hidden_dim, dropout=0.1, max_seq_length=64):
        """
        Transformer 模型，输入是id序列，支持在序列开头添加 [CLS] token，并输出 [CLS] 的嵌入。
        :param num_ids: ID 的取值范围（总数），这里是 91718
        :param embedding_dim: 嵌入维度
        :param num_heads: 多头注意力的头数
        :param num_layers: Transformer 编码器的层数
        :param hidden_dim: 前向传播隐藏层的维度
        :param dropout: Dropout 概率
        :param max_seq_length: 最大序列长度
        """
        super(Transformer_ID_WithCLS, self).__init__()
        self.num_ids = num_ids
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_ids, embedding_dim)  # 嵌入层
        self.position_embedding = nn.Embedding(max_seq_length + 1, embedding_dim)  # 位置嵌入 (+1 是为了考虑 [CLS] token)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # 支持 batch 在第一个维度
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 特殊的 [CLS] token 嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  # [CLS] token 初始化为可训练参数

    def forward(self, input_ids):
        """
        前向传播
        :param input_ids: 输入 ID，形状为 (batch_size, seq_length)
        :return: [CLS] token 的嵌入，形状为 (batch_size, embedding_dim)
        """
        batch_size, seq_length = input_ids.size()

        # ID 嵌入
        embedded = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)

        # 位置嵌入
        positions = torch.arange(0, seq_length + 1, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length + 1)
        position_embedded = self.position_embedding(positions)  # (batch_size, seq_length + 1, embedding_dim)

        # [CLS] token 嵌入
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embedding_dim)

        # 拼接 [CLS] token 和输入嵌入
        x = torch.cat([cls_token, embedded], dim=1)  # (batch_size, seq_length + 1, embedding_dim)

        # 加和位置嵌入
        x = x + position_embedded  # (batch_size, seq_length + 1, embedding_dim)

        # Transformer 编码器
        x = self.transformer_encoder(x)  # (batch_size, seq_length + 1, embedding_dim)

        # 输出 [CLS] token 的嵌入
        cls_embedding = x[:, 0, :]  # (batch_size, embedding_dim)
        return cls_embedding


class Transformer_Tags_WithCLS(nn.Module):
    def __init__(self, num_ids, embedding_dim, num_heads, num_layers, hidden_dim, dropout=0.1, max_seq_length=64):
        """
        带有 CLS 标记的 Transformer 模型，用于处理输入形状为 (batch_size, seq_length, 5) 的数据。
        :param num_ids: ID 的总数
        :param embedding_dim: 嵌入维度
        :param num_heads: 多头注意力的头数
        :param num_layers: Transformer 编码器层数
        :param hidden_dim: 前向传播隐藏层的维度
        :param dropout: Dropout 概率
        :param max_seq_length: 最大序列长度
        """
        super(Transformer_Tags_WithCLS, self).__init__()
        self.num_ids = num_ids
        self.embedding_dim = embedding_dim

        # 嵌入层
        self.item_embedding = nn.Embedding(num_ids, embedding_dim)

        # CLS 标记的嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_length + 1, embedding_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids):
        """
        前向传播
        :param input_ids: 输入 ID，形状为 (batch_size, seq_length, 5)
        :return: 序列的整体表示或预测分数
        """
        batch_size, seq_length, num_ids_per_array = input_ids.size()

        # Step 1: 对每个数组的 5 个 ID 进行嵌入
        embedded = self.item_embedding(input_ids)  # (batch_size, seq_length, 5, embedding_dim)

        # Step 2: 对数组中的 5 个嵌入进行聚合  #### 
        array_representation = embedded.mean(dim=2)  # (batch_size, seq_length, embedding_dim)

        # Step 3: 添加 CLS 标记
        # 初始化 CLS 标记的嵌入
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embedding_dim)
        sequence_with_cls = torch.cat([cls_token, array_representation], dim=1)  # (batch_size, seq_length + 1, embedding_dim)

        # Step 4: 添加位置嵌入
        positions = torch.arange(0, seq_length + 1, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length + 1)
        pos_embedded = self.position_embedding(positions)  # (batch_size, seq_length + 1, embedding_dim)
        sequence_representation = sequence_with_cls + pos_embedded  # (batch_size, seq_length + 1, embedding_dim)

        # Step 5: 输入 Transformer 编码器
        encoded_sequence = self.transformer_encoder(sequence_representation)  # (batch_size, seq_length + 1, embedding_dim)

        # Step 6: 提取 CLS 标记的表示
        cls_representation = encoded_sequence[:, 0, :]  # (batch_size, embedding_dim)

        return cls_representation

class Transformer_Embedding_WithCLS(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, hidden_dim, dropout=0.1, max_seq_length=64):
        """
        带有 CLS 标记的 Transformer 模型，用于处理输入形状为 (batch_size, seq_length, embedding_dim) 的嵌入数据。
        :param embedding_dim: 输入嵌入的维度
        :param num_heads: 多头注意力的头数
        :param num_layers: Transformer 编码器层数
        :param hidden_dim: 前向传播隐藏层的维度
        :param dropout: Dropout 概率
        :param max_seq_length: 最大序列长度
        """
        super(Transformer_Embedding_WithCLS, self).__init__()
        self.embedding_dim = embedding_dim

        # CLS 标记的嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_length + 1, embedding_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        

    def forward(self, x):
        """
        前向传播
        :param x: 输入嵌入，形状为 (batch_size, seq_length, embedding_dim)
        :return: 序列的整体表示或预测分数
        """
        batch_size, seq_length, embedding_dim = x.size()

        # Step 1: 添加 CLS 标记
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embedding_dim)
        x_with_cls = torch.cat([cls_token, x], dim=1)  # (batch_size, seq_length + 1, embedding_dim)

        # Step 2: 添加位置嵌入
        positions = torch.arange(0, seq_length + 1, device=x.device).unsqueeze(0).expand(batch_size, seq_length + 1)
        pos_embedded = self.position_embedding(positions)  # (batch_size, seq_length + 1, embedding_dim)
        x_with_pos = x_with_cls + pos_embedded  # (batch_size, seq_length + 1, embedding_dim)

        # Step 3: 输入 Transformer 编码器
        encoded_sequence = self.transformer_encoder(x_with_pos)  # (batch_size, seq_length + 1, embedding_dim)

        # Step 4: 提取 CLS 标记的表示
        cls_representation = encoded_sequence[:, 0, :]  # (batch_size, embedding_dim)

        return cls_representation



class MyTransformer(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="MyTransformer", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 din_use_softmax=False,
                 accumulation_steps=1,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(MyTransformer, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        if isinstance(dnn_activations, str) and dnn_activations.lower() == "dice":
            dnn_activations = [Dice(units) for units in dnn_hidden_units]
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        print("embedding_dim",embedding_dim)
        print('feature_map',feature_map)
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 

        print('item_info_dim',self.item_info_dim)
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.attention_layers = DIN_Attention(
            self.item_info_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            use_softmax=din_use_softmax
        )
        # ------------------ 我的模型 -------------------- # 
        self.tags_num = 5
        self.trans_attention = TransformerModel(256, 4, 2, 256, 0.1)

        self.like_emb_layer = nn.Embedding(num_embeddings=11, embedding_dim=32)
        self.view_emb_layer = nn.Embedding(num_embeddings=11, embedding_dim=32) 
        self.item_id_seq_transformer = Transformer_ID_WithCLS(
            num_ids = 91718,
            embedding_dim = 32,
            num_heads= 2,
            num_layers = 3,
            hidden_dim = 512,
            dropout = 0.1,
            max_seq_length = 64+1 # 65
        )
        self.item_tags_seq_transformer = Transformer_Tags_WithCLS(
            num_ids = 11740,
            embedding_dim = 32,
            num_heads= 2,
            num_layers = 3,
            hidden_dim = 512,
            dropout = 0.1,
            max_seq_length = 64+1 # 65
        )
        self.item_emb_seq_transformer = Transformer_Embedding_WithCLS(
            embedding_dim=128,
            num_heads=2,
            num_layers=3,
            hidden_dim=512,
            dropout=0.1,
            max_seq_length=64+1
        )
        input_dim = 448-128
        # ------------------------------------------------ # 
        # input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim

        # 
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        
        self.reset_parameters()
        self.model_to_device()
        
    def forward(self,inputs):
        return self.forward_v2(inputs)

    def forward_v1_transformer(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        print("------batch_dict------")
        print(batch_dict)
        # likes_level: torch.Size([8192])
        # views_level: torch.Size([8192])
        # 遍历字典并打印每个张量的 shape
        for key, value in batch_dict.items():
            print(f"{key}: {value.shape}")
        print("------item_dict------")
        print(item_dict)
        for key, value in item_dict.items():
            print(f"{key}: {value.shape}")
            # item_id: torch.Size([532480])
            # item_tags: torch.Size([532480, 5])
            # item_emb_d128: torch.Size([532480, 128])
        print("------mask------")
        print(mask.shape)# torch.Size([8192, 64])

        emb_list = []
        if batch_dict: # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True) # feature_emb torch.Size([8192, 128])
            emb_list.append(feature_emb) # emb_list 1
            print("feature_emb",feature_emb.shape)
 
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=False)  # item_feat_emb: torch.Size([532480, 256])
        print(item_feat_emb)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)# item_feat_emb: torch.Size([532480, 256]) -> torch.Size([8192, 65, 256])
        target_emb = item_feat_emb[:, -1, :] # target_emb torch.Size([8192, 256])
        sequence_emb = item_feat_emb[:, 0:-1, :] # sequence_emb torch.Size([8192, 64, 256])
         
        # 获取前 128 维
        # sequence_emb_before = sequence_emb[:, :, :128]  # [8192, 64, 128]
        # # 获取后 128 维
        # sequence_emb_after = sequence_emb[:, :, 128:]  # [8192, 64, 128]
        # 打印结果形状
        # print("sequence_emb_before shape:", sequence_emb_before)
        # print("sequence_emb_after shape:", sequence_emb_after)
     
        # pooling_emb = self.attention_layers(target_emb, sequence_emb, mask) # 这里可以修改
        pooling_emb = self.trans_attention(sequence_emb, target_emb, mask) # pooling_emb torch.Size([8192, 256])
        emb_list += [target_emb, pooling_emb] # emb_list 1 -> 3

        for v in emb_list:
            print(v.shape)
        # feature_emb torch.Size([8192, 128])
        # target_emb torch.Size([8192, 256])
        # pooling_emb torch.Size([8192, 256])
        
        feature_emb = torch.cat(emb_list, dim=-1)# feature_emb torch.Size([8192, 640])
        y_pred = self.dnn(feature_emb)# y_pred torch.Size([8192, 1])
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_v2(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        
        # print("------batch_dict------")
        # likes_level: torch.Size([8192])
        # views_level: torch.Size([8192])
        batch_size = mask.shape[0] # batchsize, 64
        
        likes_level_x = batch_dict['likes_level']
        views_level_x = batch_dict['views_level']
        # likes_level_emb = self.like_emb_layer(likes_level_x) # b, emb_32
        # views_level_emb = self.view_emb_layer(views_level_x) # b, emb_32
        
        item_id_seq = item_dict['item_id'].view(batch_size, -1) # b, seq+1
        target_id = item_id_seq[:, -1] # b
        sequence_id = item_id_seq[:, 0:-1] # b, seq
        
        item_tags_seq = item_dict['item_tags'].view(batch_size, -1, self.tags_num) # b, seq+1, 5
        target_tags = item_tags_seq[:, -1] # b, 5
        sequence_tags = item_tags_seq[:, 0:-1] # b, seq, 5
        
        item_emb_d128_seq = item_dict['item_emb_d128'].view(batch_size, -1, 128) # b, seq+1, emb_128
        item_emb_d128_seq = item_emb_d128_seq.float() #  转换为 float32
        target_emb_d128 = item_emb_d128_seq[:, -1, :] # b,  emb_128
        sequence_emb_d128 = item_emb_d128_seq[:, 0:-1, :] #b, seq, emb_128

        likes_level_emb = self.like_emb_layer(likes_level_x) # b, emb_32
        views_level_emb = self.view_emb_layer(views_level_x) # b, emb_32

        sequence_id2emb = self.item_id_seq_transformer(item_id_seq) # b, emb_128
        sequence_tags2emb = self.item_tags_seq_transformer(item_tags_seq) # b, emb_128
        # print(item_emb_d128_seq.dtype)
        sequence_emb2emb = self.item_emb_seq_transformer(item_emb_d128_seq)# b, emb_128
        
        emb_list = [likes_level_emb, views_level_emb,sequence_tags2emb,sequence_emb2emb]
        
        
        # emb_list = []
        # if batch_dict: # not empty
        #     feature_emb = self.embedding_layer(batch_dict, flatten_emb=True) # feature_emb torch.Size([8192, 128])
        #     emb_list.append(feature_emb) # emb_list 1
        #     print("feature_emb",feature_emb.shape)
 
        # item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)  # item_feat_emb: torch.Size([532480, 256])
        # print(item_feat_emb)
        # batch_size = mask.shape[0]
        # item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)# item_feat_emb: torch.Size([532480, 256]) -> torch.Size([8192, 65, 256])
        # target_emb = item_feat_emb[:, -1, :] # target_emb torch.Size([8192, 256])
        # sequence_emb = item_feat_emb[:, 0:-1, :] # sequence_emb torch.Size([8192, 64, 256])
         
        # # 获取前 128 维
        # # sequence_emb_before = sequence_emb[:, :, :128]  # [8192, 64, 128]
        # # # 获取后 128 维
        # # sequence_emb_after = sequence_emb[:, :, 128:]  # [8192, 64, 128]
        # # 打印结果形状
        # # print("sequence_emb_before shape:", sequence_emb_before)
        # # print("sequence_emb_after shape:", sequence_emb_after)
     
        # # pooling_emb = self.attention_layers(target_emb, sequence_emb, mask) # 这里可以修改
        # pooling_emb = self.trans_attention(sequence_emb, target_emb, mask) # pooling_emb torch.Size([8192, 256])
        # emb_list += [target_emb, pooling_emb] # emb_list 1 -> 3

        # for v in emb_list:
        #     print(v.shape)
        # # feature_emb torch.Size([8192, 128])
        # # target_emb torch.Size([8192, 256])
        # # pooling_emb torch.Size([8192, 256])
        
        feature_emb = torch.cat(emb_list, dim=-1)# feature_emb torch.Size([8192, 640])
        # print(feature_emb.shape)
        y_pred = self.dnn(feature_emb)# y_pred torch.Size([8192, 1])
        return_dict = {"y_pred": y_pred}
        return return_dict


    
    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss
