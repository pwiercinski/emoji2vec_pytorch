import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, input_size, num_emojis, dropout):
        super().__init__()
        self.V = torch.nn.Parameter(torch.empty(num_emojis, input_size).uniform_(-0.1, 0.1))
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, x, emoji_ids):
        weight = self.V[emoji_ids]
        weight = self.dropout(weight)
        x = torch.sum(torch.mul(weight, x), 1)
        x = torch.sigmoid(x)
        return x
