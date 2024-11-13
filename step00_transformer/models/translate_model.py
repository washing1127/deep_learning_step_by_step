'''
Author: washing1127
Date: 2024-11-09 23:34:30
LastEditors: washing1127
LastEditTime: 2024-11-13 10:46:43
FilePath: /llms_step_by_step/step00_transformer/models/translate_model.py
Description: 
'''
import math
import torch
import torch.nn as nn

from torch.nn import Transformer
from .my_transformer import MyTransformer

from utils import PositionEncoding

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class TranslateModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, max_seq_len=128, dropout=0.1, need_mask=True, which_model=""):
        super().__init__()
        # TODO: 这里尝试过直接以 nn.Embedding 代替 TokenEmbedding，并在 forward 的时候乘以 math.sqrt(d_model)。但是失败了，有待后续详细研究；
        self.src_embedder = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedder = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionEncoding(d_model, dropout, max_seq_len)
        if which_model.lower() == "official":
            self.transformer = Transformer(batch_first=True)
        elif which_model.lower() == "myself":
            self.transformer = MyTransformer()
        else:
            raise ValueError("param: which_model should in ['official', 'myself'].")
        self.classifier = nn.Linear(d_model, tgt_vocab_size)
        self.need_mask = need_mask
        self.d_model = d_model
        self._reset_parameters()  # 参数初始化为高斯分布

    def forward(self, src, tgt):
        src_embed = self.src_embedder(src)
        src_embed = self.pos_encoder(src_embed)
        tgt_embed = self.tgt_embedder(tgt)
        tgt_embed = self.pos_encoder(tgt_embed)

        tgt_mask = self._create_mask(tgt)

        outs = self.transformer(
            src=src_embed,
            tgt=tgt_embed,
            tgt_mask=tgt_mask
        )

        logits = self.classifier(outs)
        return logits

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_mask(self, tgt):
        device = tgt.device
        tgt_seq_len = tgt.shape[1]
        tgt_mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float("-inf")).masked_fill(tgt_mask == 1, float(0.0))
        return tgt_mask
