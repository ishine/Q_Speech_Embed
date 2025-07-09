import torch
from utils import SymQuant8bit, LoggerUnit
from model import DualGateBN_Q

if __name__ == "__main__":
    logger = LoggerUnit("DualGateBN_Q_Test").get_logger()
    for qscale in [1.0, 0.25]:
        logger.info(f"Testing DualGateBN_Q with quantscale={qscale}.")
        quant = SymQuant8bit(quantscale=qscale)
        block = DualGateBN_Q(7, 14, 3, 6, 7, quantizer=quant)

        x = torch.randn(1, 7, 12, 98)
        out = block(x)
        logger.info(f"Output shape: {out.shape}.")
