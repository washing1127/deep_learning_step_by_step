'''
Author: washing1127
Date: 2024-11-09 23:34:30
LastEditors: washing1127
LastEditTime: 2024-11-10 00:08:23
FilePath: /llms_step_by_step/step00_transformer/models/translate_model.py
Description: 
'''
import math
import torch
import torch.nn as nn

from torch.nn import Transformer

from data import PAD_IDX
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
            self.transformer = Transformer()
        elif which_model.lower() == "myself":
            print("myself 还没实现")
            exit()
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

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = self._create_mask(src, tgt)

        outs = self.transformer(
            src=src_embed,
            tgt=tgt_embed,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        logits = self.classifier(outs)
        return logits

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_mask(self, src, tgt):
        device = src.device
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
        tgt_mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float("-inf")).masked_fill(tgt_mask == 1, float(0.0))
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
        src_pad_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_pad_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_pad_mask, tgt_pad_mask

