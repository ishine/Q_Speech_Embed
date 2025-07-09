from utils import LoggerUnit, SymQuant8bit

from model import Linear_Q
import torch


logger = LoggerUnit("Linear_Q_Test").get_logger()

if __name__ == "__main__":
    
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
