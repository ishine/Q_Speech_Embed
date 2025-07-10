# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils import SymQuant8bit, LoggerUnit


# class BatchNorm2D_Q(nn.BatchNorm2d):
#     def __init__(self, num_features, quantizer=None):
#         super().__init__(num_features)
#         self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

#     def forward(self, x):
#         if self.quantizer.enabled:
#             x_q, x_scale = self.quantizer.quantize(x)
#             x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

#             w_q, w_scale = self.quantizer.quantize(self.weight)
#             w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)

#             b_q, b_scale = self.quantizer.quantize(self.bias)
#             b_fq = self.quantizer.apply_fake_quant(b_q, b_scale, self.bias)
#         else:
#             x_fq = x
#             w_fq = self.weight
#             b_fq = self.bias

#         return F.batch_norm(x_fq, self.running_mean, self.running_var,
#                             w_fq, b_fq, self.training, self.momentum, self.eps)

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SymQuant8bit

class BatchNorm2D_Q(nn.BatchNorm2d):
    def __init__(self, num_features, quantizer=None):
        super().__init__(num_features)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

    def forward(self, x):
        if self.quantizer.enabled:
            x_q, x_scale = self.quantizer.quantize(x)
            x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

            w_q, w_scale = self.quantizer.quantize(self.weight)
            w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)

            b_q, b_scale = self.quantizer.quantize(self.bias)
            b_fq = self.quantizer.apply_fake_quant(b_q, b_scale, self.bias)
        else:
            x_fq = x
            w_fq = self.weight
            b_fq = self.bias

        return F.batch_norm(x_fq, self.running_mean, self.running_var,
                            w_fq, b_fq, self.training, self.momentum, self.eps)
