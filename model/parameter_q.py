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


if __name__ == "__main__":
    logger = LoggerUnit("Parameter_Q_Test").get_logger()
    torch.manual_seed(0)

    x = torch.tensor([2.85])
    qp = Parameter_Q(torch.tensor([0.937]))

    y = qp(x)

    logger.info(f"Input: {x.item():.4f}.")
    logger.info(f"Param (learnable): {qp.param.item():.4f}.")
    logger.info(f"Output (quantized product): {y.item():.4f}.")
