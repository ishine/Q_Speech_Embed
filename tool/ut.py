# import torch
# import numpy as np
# from model import BitGateNet

# net = BitGateNet(num_classes=8, q_en=True)
# net.load_state_dict(torch.load("checkpoints/b_m_0.8298.pth"), strict=False)
# net.eval()

# # Input (should match C input)
# x = torch.from_numpy(np.load("infer_c/stub_inp.npy"))
# if x.ndim == 3:
#     x = x.unsqueeze(0)
# xq = x.flatten().to(torch.int8).numpy()
# print("PY: First 10 input values (int8):", xq[:10].tolist())

# # Weights (should match C weights for conv1)
# w = net.conv1.weight.detach().cpu().flatten().to(torch.int8).numpy()
# print("PY: First 10 conv1 weights (int8):", w[:10].tolist())

# # Scale (if layer uses, otherwise skip)
# if hasattr(net.conv1, 'scale'):
#     s = net.conv1.scale.detach().cpu().numpy()
#     print("PY: First 10 conv1 scale values:", s.flatten()[:10].tolist())
import torch
from model import BitGateNet
from utils.symquant8bit import SymQuant8bit

net = BitGateNet(num_classes=8, q_en=False)  # doesn't matter for loading.
ckpt = torch.load("checkpoints/b_m_0.8298.pth", map_location="cpu")
net.load_state_dict(ckpt, strict=False)

# Grab float weights from checkpoint (example: conv1)
float_weights = net.conv1.weight.data
quant = SymQuant8bit()
q_x, scale = quant.quantize(float_weights)

print("First 10 quantized weights:", q_x.flatten()[:10].tolist())
print("First 10 scale values:", scale.flatten()[:10].tolist())
