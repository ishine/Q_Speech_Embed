import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SymQuant8bit

class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_mode='zeros',
                 quantizer=None):
        super().__init__(in_channels, out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)

        self.quantizer = quantizer or SymQuant8bit(group_dim=1, group_size=1)
        self.groups = groups

    def forward(self, x):
        # Quantize + fake-quant input activations.
        x_q, x_scale = self.quantizer.quantize(x)
        x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

        # Quantize + fake-quant weights.
        w_q, w_scale = self.quantizer.quantize(self.weight)
        w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)

        # Bias handling.(if you want)
        bias = self.bias if self.bias is not None else None

        return F.conv2d(x_fq, w_fq, bias=bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)
