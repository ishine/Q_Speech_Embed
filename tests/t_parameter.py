import torch
from utils import LoggerUnit
from model import Parameter_Q

logger = LoggerUnit("Parameter_Q_Test").get_logger()
if __name__ == "__main__":
    
    torch.manual_seed(0)

    x = torch.tensor([2.85])
    qp = Parameter_Q(torch.tensor([0.937]))

    y = qp(x)

    logger.info(f"Input: {x.item():.4f}.")
    logger.info(f"Param (learnable): {qp.param.item():.4f}.")
    logger.info(f"Output (quantized product): {y.item():.4f}.")
