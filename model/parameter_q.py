# import torch
# import torch.nn as nn
# from utils import SymQuant8bit, LoggerUnit

# class Parameter_Q(nn.Module):
#     """
#     Learnable parameter module with symmetric quantization.
#     """
#     def __init__(self, val, quantizer=None):
#         super().__init__()
#         self.param = nn.Parameter(val)
#         self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

#     def forward(self, x):
#         if self.quantizer.enabled:
#             q_p, scale_p = self.quantizer.quantize(self.param)
#             param_fq = self.quantizer.apply_fake_quant(q_p, scale_p, self.param)

#             q_x, scale_x = self.quantizer.quantize(x)
#             x_fq = self.quantizer.apply_fake_quant(q_x, scale_x, x)
#         else:
#             param_fq = self.param
#             x_fq = x

#         return x_fq * param_fq

import torch
import torch.nn as nn
from utils import SymQuant8bit

class Parameter_Q(nn.Module):
    def __init__(self, val, quantizer=None):
        super().__init__()
        self.param = nn.Parameter(val)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

    def forward(self, x=None):
        return self.param
    # def forward(self, x):
    #     if self.quantizer.enabled:
    #         q_p, scale_p = self.quantizer.quantize(self.param)
    #         param_fq = self.quantizer.apply_fake_quant(q_p, scale_p, self.param)

    #         q_x, scale_x = self.quantizer.quantize(x)
    #         x_fq = self.quantizer.apply_fake_quant(q_x, scale_x, x)
    #     else:
    #         param_fq = self.param
    #         x_fq = x

    #     return x_fq * param_fq


