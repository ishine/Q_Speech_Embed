import torch.nn as nn
from .gateresidual_q import GateResidual_Q
from .batchnorm2d_q import BatchNorm2D_Q
from utils import SymQuant8bit, LoggerUnit

class DualGateBN_Q(nn.Module):
    def __init__(self, in_channels, mid_channels_0, out_channels_0, mid_channels_1, out_channels_1,
                 quantizer=None, test=0):
        super(DualGateBN_Q, self).__init__()

        self.quantizer = quantizer or SymQuant8bit(quantscale=0.25)

        self.layer1 = GateResidual_Q(in_channels, mid_channels_0, out_channels_0,
                                    quantizer=self.quantizer)
        self.layer2 = BatchNorm2D_Q(in_channels, quantizer=self.quantizer)

        self.layer3 = GateResidual_Q(in_channels, mid_channels_1, out_channels_1,
                                    quantizer=self.quantizer)
        self.layer4 = BatchNorm2D_Q(in_channels, quantizer=self.quantizer)

        self.test = test

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out += x
        return out

