import torch
from utils import LoggerUnit, SymQuant8bit

logger = LoggerUnit("SymQuantTest").get_logger()

if __name__ == "__main__":
   
    
    torch.manual_seed(0)

    quant = SymQuant8bit(quantscale=1.0)

    def test_tensor(name, x):
        q, scale = quant.quantize(x, verbose=True)
        x_hat = quant.dequantize(q, scale)
        err = (x - x_hat).abs().max().item()

        logger.info(f"{name} | shape: {tuple(x.shape)}")
        logger.info(f"- scale: {scale}")
        logger.info(f"- quantized range: {q.min().item()} to {q.max().item()}")
        logger.info(f"- dequant error (max abs diff): {err:.6f}")
        print("")

    # Scalar..
    test_tensor("Scalar", torch.tensor(0.75))

    # 1D Vector
    test_tensor("1D vector", torch.tensor([0.1, 0.3, 0.9]))

    # ND weights (ex: conv).
    test_tensor("Conv weights", torch.randn(8, 3, 3, 3))

    # ND uneven groups.
    test_tensor("Uneven groups", torch.randn(7, 5, 3, 3))

    # All zeros.
    test_tensor("All zeros", torch.zeros(4, 4))

    # Outlier-heavy tensor.
    test_tensor("Outliers", torch.tensor([0.1, 0.2, 900.0, 0.3]))
