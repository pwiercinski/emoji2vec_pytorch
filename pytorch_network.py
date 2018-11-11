import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, input_size, output_size, num_emojis, dropout):
        super().__init__()
        self.V = torch.nn.Parameter(torch.empty(num_emojis, output_size).uniform_(-0.1, 0.1))
        self.dropout = torch.nn.Dropout(p=dropout)
        if not input_size == output_size:
            self.is_proj = True
            self.W = torch.nn.Parameter(torch.empty(input_size, output_size).uniform_(-0.1, 0.1))
            self.tanh = torch.nn.Tanh()
        else:
            self.is_proj = False

    def forward(self, x, emoji_ids):
        if self.is_proj:
            proj = torch.mm(x, self.W)
            x = self.tanh(proj)
        weights = self.V[emoji_ids]
        weights = self.dropout(weights)
        x = torch.sum(torch.mul(weights, x), 1)
        x = torch.sigmoid(x)

        return x

    def project_embeddings(self, embeddings_array):
        return self.tanh(torch.mm(torch.Tensor(embeddings_array), self.W)).detach().numpy()
