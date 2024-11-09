import math

import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)
    def forward(self, x):  # [seq_len, batch_size, d_model]
        return self.dropout(x + self.pe[: x.size(0), :])