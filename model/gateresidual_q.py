import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv2d_q import Conv2d_Q
from .parameter_q import Parameter_Q
from utils import SymQuant8bit, LoggerUnit


class GateResidual_Q(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, quantizer=None):
        super().__init__()

        self.quantizer = quantizer or SymQuant8bit(quantscale=1.0)
        self.logger = LoggerUnit("GateResidual").get_logger()

        self.condense = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_conv = Conv2d_Q(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, quantizer=self.quantizer)
        self.pointwise_conv = Conv2d_Q(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, quantizer=self.quantizer)
        self.expand_conv = Conv2d_Q(out_channels, in_channels, kernel_size=1, stride=1, padding=0, quantizer=self.quantizer)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.scale = Parameter_Q(torch.tensor([[1.0]]), quantizer=self.quantizer)

    def forward(self, x):
        residual = x

        q = self.condense(x)
        k = F.relu(self.group_conv(q))
        k = F.relu(self.pointwise_conv(k))

        #self.logger.debug(f"[GateResidual] Before upsample: k shape = {k.shape}")
        a = self.upsample(k)
        # self.logger.debug(f"[GateResidual] After upsample:  a shape = {a.shape}")
        a = self.expand_conv(a)

        s = torch.sigmoid(a)
        #v = residual * s * self.scale(residual)
        v = residual * s * self.scale()
        return v + residual
