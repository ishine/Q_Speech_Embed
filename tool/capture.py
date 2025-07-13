import torch
import numpy as np
from model import BitGateNet

import torch
import numpy as np
from model import BitGateNet

# Replace this with the actual scale value from weights.c (for conv1)
conv1_scale = 0.150872  # real value from weights.

net = BitGateNet(num_classes=8, q_en=True)
net.load_state_dict(torch.load("checkpoints/b_m_0.8298.pth"), strict=False)
net.eval()

x = torch.from_numpy(np.load("infer_c/stub_inp.npy"))
if x.ndim == 3:
    x = x.unsqueeze(0)

# Global to capture output.
conv1_out = None

def conv1_hook(module, inp, outp):
    global conv1_out
    conv1_out = outp.detach().cpu().numpy()
    # Remove batch if needed
    if conv1_out.shape[0] == 1:
        conv1_out = conv1_out[0]
    # Quantize like in C
    conv1_q = np.round(conv1_out / conv1_scale).clip(-127,127).astype(np.int8)
    print("PyTorch conv1 (float)   first 10:", conv1_out.flatten()[:10].tolist())
    print("PyTorch conv1 (int8Q)   first 10:", conv1_q.flatten()[:10].tolist())

net.conv1.register_forward_hook(conv1_hook)

with torch.no_grad():
    _ = net(x)

# Load C dump for conv1.
def read_conv1_dump():
    with open("infer_c/dump_conv1.bin", "rb") as f:
        dims = int.from_bytes(f.read(1), 'little')
        shape = list(f.read(dims))
        size = np.prod(shape)
        data = np.frombuffer(f.read(), dtype=np.int8)[:size]
        return data.reshape(shape)

c_conv1 = read_conv1_dump()
print("C conv1 first 10:", c_conv1.flatten()[:10].tolist())

# want to compare? =>
conv1_q = np.round(conv1_out / conv1_scale).clip(-127,127).astype(np.int8)
mse = np.mean((c_conv1.flatten() - conv1_q.flatten()) ** 2)
print("MSE PyTorch-vs-C:", mse)


#####################################################################


def read_conv1():
    with open("infer_c/dump_conv1.bin", "rb") as f:
        dims = int.from_bytes(f.read(1), 'little')
        shape = list(f.read(dims))
        size = np.prod(shape)
        data = np.frombuffer(f.read(), dtype=np.int8)[:size]
        print("C conv1 first 10:", data.flatten()[:10].tolist())

read_conv1()
