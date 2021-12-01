from __future__ import annotations
import torch
import hyperparams as hp
from torch import nn as nn
from torch.nn import functional


class Attention(nn.Module):
    """
    Base attention module, don't use directly. Use one of its subclasses instead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def score(self, hidden, encoder_output):
        raise NotImplementedError()

    def forward(self, hidden, encoder_outputs):
        alphas = self.score(hidden, encoder_outputs)
        return functional.softmax(alphas.t(), dim=1).unsqueeze(1)

    @classmethod
    def build(cls, attention_type: str) -> Attention:
        """
        Returns a new instance of the attention module specified by attention_type.
        """
        if attention_type == 'dot':
            return DotAttention()
        elif attention_type == 'mul':
            return MultiplicativeAttention()
        elif attention_type == 'add':
            return AdditiveAttention()
        else:
            raise ValueError(f'Unknown attention type: {attention_type}')


class DotAttention(Attention):
    """
    Dot-product attention.
    """
    def score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)


class MultiplicativeAttention(Attention):
    """
    Multiplicative (bi-linear) attention.
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(hp.HIDDEN_LAYER_DIM, hp.HIDDEN_LAYER_DIM)

    def score(self, hidden, encoder_output):
        alphas = self.attn(encoder_output)
        return torch.sum(hidden * alphas, dim=2)


class AdditiveAttention(Attention):
    """
    Additive (essentially MLP) attention.
    """
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(hp.HIDDEN_LAYER_DIM * 2, hp.HIDDEN_LAYER_DIM)
        self.v = nn.Parameter(torch.FloatTensor(hp.HIDDEN_LAYER_DIM))

    def score(self, hidden, encoder_output):
        alphas = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * alphas, dim=2)
