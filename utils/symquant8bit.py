import torch
import torch.nn as nn
from .log_tools import LoggerUnit

class SymQuant8bit:
    def __init__(self, group_dim=1, group_size=1, eps=1e-5, quantscale=1.0, enabled=True):
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
        self.enabled = enabled
        self.logger = LoggerUnit("SymQuant8bit").get_logger()

    def quantize(self, x, verbose=False):
        if not self.enabled:
            return x, 1
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
        if not self.enabled:
            return x
        return (x + (q_x / scale - x).detach())
    
    # def apply_fake_quant(self, q_x, scale, x):
    #     if not self.enabled:
    #         return x
    #     out = x + (q_x / scale - x).detach()
    #     self.logger.debug(f"[SymQuant8bit] Fake quant out: {out.shape}, is_leaf={out.is_leaf}")
    #     return out

    def __str__(self):
        return f"SymQuant8bit(group_dim={self.group_dim}, group_size={self.group_size}, scale={self.quantscale})"

