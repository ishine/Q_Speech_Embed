import torch
from model import GateResidual_Q
from utils import LoggerUnit, SymQuant8bit


logger = LoggerUnit("GateResidual_Test").get_logger()

if __name__ == "__main__":
    
    torch.manual_seed(0)

    for qscale in [1.0, 0.25]:
        logger.info(f"Testing QGateResidual with quantscale={qscale}.")
        quant = SymQuant8bit(quantscale=qscale)
        block = GateResidual_Q(7, 14, 3, quantizer=quant)

        x = torch.randn(1, 7, 12, 98)
        out = block(x)

        logger.info(f"Output shape: {out.shape}.")
        wq, _ = quant.quantize(block.expand_conv.weight)
        logger.info(f"Quantized expand_conv weight range: {wq.min().item()} to {wq.max().item()}.")
