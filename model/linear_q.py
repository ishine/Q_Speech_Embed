import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SymQuant8bit

class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quantizer=None):
        super().__init__(in_features, out_features, bias=bias)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

    def forward(self, x):
        if self.quantizer.enabled:
            x_q, x_scale = self.quantizer.quantize(x)
            x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

            w_q, w_scale = self.quantizer.quantize(self.weight)
            w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)
        else:
            x_fq = x
            w_fq = self.weight

        return F.linear(x_fq, w_fq, self.bias)