import torch
import torch.nn as nn
from utils import SymQuant8bit, LoggerUnit

class Parameter_Q(nn.Module):
    """
    Learnable parameter module with symmetric quantization.
    """
    def __init__(self, val, quantizer=None):
        super().__init__()
        self.param = nn.Parameter(val)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

    def forward(self, x):
        # Quantize + fake-quant the parameter.
        q_p, scale_p = self.quantizer.quantize(self.param)
        param_fq = self.quantizer.apply_fake_quant(q_p, scale_p, self.param)

        # Quantize + fake-quant the input.
        q_x, scale_x = self.quantizer.quantize(x)
        x_fq = self.quantizer.apply_fake_quant(q_x, scale_x, x)

        return x_fq * param_fq

