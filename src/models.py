"""Neural model definitions for CyberRAX-CP and baselines."""
from __future__ import annotations
import torch
from torch import nn

class GRUAttention(nn.Module):
    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.attn_hidden = nn.Linear(hidden_size, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention: bool = False):
        sequence, _ = self.gru(x)
        scores = self.attn_score(torch.tanh(self.attn_hidden(sequence))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        logits = self.head(self.dropout(context)).squeeze(-1)
        if return_attention:
            return logits, weights
        return logits

class TransformerDetector(nn.Module):
    def __init__(self, input_size: int = 15, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2, max_length: int = 30):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_length, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x):
        z = self.input_projection(x) + self.position[:, :x.shape[1], :]
        z = self.encoder(z)
        z = self.norm(z[:, -1, :])
        return self.head(z).squeeze(-1)
