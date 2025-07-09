import torch

from model import Conv2d_Q
from utils import SymQuant8bit, LoggerUnit

logger = LoggerUnit("Conv2D_Test").get_logger()


if __name__ == "__main__":
    x = torch.randn(1, 1, 12, 98)

    for qscale in [1.0, 0.25]:
        logger.info(f"\nTesting with quantscale={qscale}")
        quantizer = SymQuant8bit(quantscale=qscale)
        conv = Conv2d_Q(1, 4, kernel_size=3, stride=1, padding=1, quantizer=quantizer, bias=True)

        out = conv(x)
        logger.info(f"Output shape: {out.shape}")

        w_q, _ = quantizer.quantize(conv.weight)
        logger.info("Quantized weight range:", w_q.min().item(), "to", w_q.max().item())

