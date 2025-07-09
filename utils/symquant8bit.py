import torch
import torch.nn as nn
from .log_tools import LoggerUnit

class SymQuant8bit:
    def __init__(self, group_dim=1, group_size=1, eps=1e-5, quantscale=1.0):
        """
        Symmetric 8-bit quantizer with group-wise support.

        Args:
            group_dim (int): Dimension along which to group.
            group_size (int): Channels per group (1 = per-channel).
            eps (float): Min value to avoid division by zero.
            quantscale (float): Soft scaling factor (default 1.0 = full range).
        """
        self.group_dim = group_dim
        self.group_size = group_size
        self.eps = eps
        self.quantscale = quantscale
        self.logger = LoggerUnit("SymQuant8bit").get_logger()

    def quantize(self, x, verbose=False):
        x_abs = x.abs()

        # Scalar (0D tensor)
        if x.dim() == 0:
            max_val = x_abs.clamp(min=self.eps)
            scale = (127.0 * self.quantscale) / max_val
            q_x = (x * scale).round().clamp(-127, 127)
            if verbose:
                self.logger.debug(f"[0D] scale={scale.item()}")
            return q_x, scale

        # Vector (1D)
        if x.dim() == 1:
            max_val = x_abs.max().clamp(min=self.eps)
            scale = (127.0 * self.quantscale) / max_val
            q_x = (x * scale).round().clamp(-127, 127)
            if verbose:
                self.logger.debug(f"[1D] scale={scale.item()}")
            return q_x, scale

        # Grouped (ND)
        C = x.size(self.group_dim)
        G = (C + self.group_size - 1) // self.group_size
        shape = [1] * x.dim()
        shape[self.group_dim] = G

        group_ids = torch.arange(C, device=x.device) // self.group_size
        group_ids = group_ids.view([-1 if i == self.group_dim else 1 for i in range(x.dim())])

        max_vals = torch.zeros(G, device=x.device)
        for g in range(G):
            mask = (group_ids == g)
            group_vals = x_abs * mask
            max_vals[g] = group_vals.max().clamp(min=self.eps)

        scale = (127.0 * self.quantscale) / max_vals
        scale = scale.view(shape)

        q_x = (x * scale).round().clamp(-127, 127)

        if verbose:
            self.logger.debug(f"[ND] scale shape={scale.shape}, quantized min={q_x.min()}, max={q_x.max()}")

        return q_x, scale

    def dequantize(self, q_x, scale):
        return q_x / scale

    def apply_fake_quant(self, q_x, scale, x):
        return x + (q_x / scale - x).detach()

    def __str__(self):
        return f"SymQuant8bit(group_dim={self.group_dim}, group_size={self.group_size}, scale={self.quantscale})"


if __name__ == "__main__":
    from log_tools import LoggerUnit
    logger = LoggerUnit("SymQuantTest").get_logger()
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
