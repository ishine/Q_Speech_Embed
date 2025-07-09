import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SymQuant8bit, LoggerUnit


class BatchNorm2D_Q(nn.BatchNorm2d):
    def __init__(self, num_features, quantizer=None):
        super().__init__(num_features)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)

    def forward(self, x):
        # Quantize + fake-quant input.
        x_q, x_scale = self.quantizer.quantize(x)
        x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

        # Quantize + fake-quant weight (gamma).
        w_q, w_scale = self.quantizer.quantize(self.weight)
        w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)

        # Quantize + fake-quant bias (beta).
        b_q, b_scale = self.quantizer.quantize(self.bias)
        b_fq = self.quantizer.apply_fake_quant(b_q, b_scale, self.bias)

        return F.batch_norm(x_fq, self.running_mean, self.running_var,
                            w_fq, b_fq, self.training, self.momentum, self.eps)


if __name__ == "__main__":
    logger = LoggerUnit("BatchNorm2D_Q_Test").get_logger()
    torch.manual_seed(0)

    for qscale in [1.0, 0.25]:
        logger.info(f"Testing BatchNorm2D_Q with quantscale={qscale}.")

        quant = SymQuant8bit(quantscale=qscale)
        layer = BatchNorm2D_Q(4, quantizer=quant)

        # Inject non-uniform gamma values before forward pass.
        with torch.no_grad():
            layer.weight[:] = torch.tensor([0.5, 0.8, 1.2, 1.7])

        logger.info(f"Gamma before quantization: {layer.weight.data.tolist()}.")

        x = torch.randn(1, 4, 6, 49)
        out = layer(x)

        logger.info(f"Output shape: {out.shape}.")
        wq, _ = quant.quantize(layer.weight)
        logger.info(f"Quantized gamma values: {wq.tolist()}.")
        logger.info(f"Quantized gamma range: {wq.min().item()} to {wq.max().item()}.")
