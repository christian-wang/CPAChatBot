import torch
from torch import nn as nn
from torch.nn import functional


class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def score(self, hidden, encoder_output):
        raise NotImplementedError()

    def forward(self, hidden, encoder_outputs):
        alphas = self.score(hidden, encoder_outputs)
        return functional.softmax(alphas.t(), dim=1).unsqueeze(1)


class DotAttention(Attention):
    def score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)


class GeneralAttention(Attention):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)


class ConcatAttention(Attention):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)
