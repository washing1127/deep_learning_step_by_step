'''
Author: washing1127
Date: 2024-11-10 21:34:30
LastEditors: washing1127
LastEditTime: 2024-11-13 10:46:43
FilePath: /llms_step_by_step/step00_transformer/models/my_transformer.py
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model, d_qkv):
        super().__init__()
        self.wq = nn.Linear(d_model, d_qkv)
        self.wk = nn.Linear(d_model, d_qkv)
        self.wv = nn.Linear(d_model, d_qkv)
        self.d_qkv = d_qkv
    def forward(self, src, tgt, mask=None):
        query = self.wq(tgt)
        key = self.wk(src)
        value = self.wv(src)
        attention_score = query @ key.transpose(1, 2)
        if mask is not None:
            assert mask.shape == torch.Size((attention_score.size(-1), attention_score.size(-1)))
            attention_score += mask
        out_value = F.softmax(attention_score * (self.d_qkv ** -0.5), dim=-1) @ value
        return out_value

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        d_qkv = d_model // n_head
        self.attentions = nn.ModuleList(
            [Attention(d_model, d_qkv) for _ in range(n_head)]
        )
        self.Wo = nn.Linear(d_model, d_model)
        self.d_model = d_model
    def forward(self, src, tgt, mask=None):
        value = torch.cat([attention(src, tgt, mask) for attention in self.attentions], dim=-1)
        return self.Wo(value)

class FFN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout1(self.relu(self.ffn1(x)))
        return self.dropout2(self.ffn2(x))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(d_model, dropout)
    def forward(self, x):
        x_norm1 = self.layer_norm1(x)
        x = x + self.dropout(self.multi_head_self_attention(x_norm1, x_norm1))
        x_norm2 = self.layer_norm2(x)
        x = x + self.ffn(x_norm2)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.masked_multi_head_self_attention = MultiHeadAttention(d_model, n_head)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = FFN(d_model)
    def forward(self, tgt, memory, mask=None):
        tgt_norm1 = self.layer_norm1(tgt)
        tgt = tgt + self.dropout1(self.masked_multi_head_self_attention(tgt_norm1, tgt_norm1, mask))
        tgt_norm2 = self.layer_norm2(tgt)
        tgt = tgt + self.dropout2(self.encoder_decoder_attention(memory, tgt_norm2))
        tgt_norm3 = self.layer_norm3(tgt)
        tgt = tgt + self.ffn(tgt_norm3)
        return tgt

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        tgt = self.layer_norm(tgt)
        return tgt

class MyTransformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, n_head, n_layers, dropout)
        self.decoder = Decoder(d_model, n_head, n_layers, dropout)
    def forward(self, src, tgt, tgt_mask=None):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, mask=tgt_mask)
        return output

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d_model = 512
    n_head = 8
    batch_size = 128
    src_seq_len = 30
    tgt_seq_len = 28

    src = torch.rand((batch_size, src_seq_len, d_model))
    tgt = torch.rand((batch_size, tgt_seq_len, d_model))

    src = src.to(device)
    tgt = tgt.to(device)
    
    # # 多头注意力块
    # attention = MultiHeadAttention(d_model, n_head)
    # output = attention(src,  tgt)
    # print(output.shape) # should be [batch_size, tgt_seq_len, d_model]  => [128, 28, 512]

    # # 单个 encoder 层
    # encoder_layer = EncoderLayer(d_model, n_head)
    # out = encoder_layer(src)
    # print(out.shape)
    
    # # encoder 模块
    # encoder = Encoder(d_model, n_head, n_layers=2)
    # out = encoder(src)
    # print(out.shape)
    
    # # 单个 decoder 层
    # decoder_layer = DecoderLayer(d_model, n_head)
    # out = decoder_layer(tgt, src)
    # print(out.shape)
    
    # # decoder 模块
    # decoder = Decoder(d_model, n_head, n_layers=2)
    # out = decoder(tgt, src)
    # print(out.shape)
    
    # # transformer
    transformer = MyTransformer(d_model, n_head, n_layers=2)
    transformer = transformer.to(device)
    out = transformer(src, tgt)
    print(out.shape)