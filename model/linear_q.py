import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SymQuant8bit, LoggerUnit

class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quantizer=None):
        super().__init__(in_features, out_features, bias=bias)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

    def forward(self, x):
        # Quantize + fake-quant input tensor.
        x_q, x_scale = self.quantizer.quantize(x)
        x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

        # Quantize + fake-quant weights.
        w_q, w_scale = self.quantizer.quantize(self.weight)
        w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)

        return F.linear(x_fq, w_fq, self.bias)

if __name__ == "__main__":
    logger = LoggerUnit("Linear_Q_Test").get_logger()
    torch.manual_seed(0)

    for qscale in [1.0, 0.25]:
        logger.info(f"Testing Linear_Q with quantscale={qscale}.")
        quant = SymQuant8bit(quantscale=qscale)
        layer = Linear_Q(17, 8, quantizer=quant)

        x = torch.randn(1, 17)
        out = layer(x)

        logger.info(f"Output shape: {out.shape}.")
        wq, _ = quant.quantize(layer.weight)
        logger.info(f"Quantized weight range: {wq.min().item()} to {wq.max().item()}.")
