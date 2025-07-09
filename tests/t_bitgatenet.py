import torch
from model import BitGateNet


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BitGateNet(num_classes=8, quantscale=1.0)
    dummy = torch.randn(1, 1, 12, 98)
    out = model(dummy)
    print("BitGateNet output shape:", out.shape)
    print("Model loaded successfully with", sum(p.numel() for p in model.parameters()), "params")
