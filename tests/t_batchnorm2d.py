from model import BatchNorm2D_Q
from utils import SymQuant8bit, LoggerUnit
import torch

logger = LoggerUnit("BatchNorm2D_Q_Test").get_logger()
if __name__ == "__main__":
    
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
