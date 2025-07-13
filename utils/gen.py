# torch_dump.py – dump all activations (intermediate tensors) for C comparison
import os, torch, numpy as np

# Conf
os.makedirs("out", exist_ok=True)
CKPT    = "checkpoints/b_m_0.8298.pth"
DEVICE  = "cpu"
USE_QAT = True

from model import BitGateNet
net  = BitGateNet(num_classes=8, quantscale=1.0, q_en=USE_QAT).to(DEVICE)
net.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
net.eval()

def dump(tag, tensor):
    fn = f"out/ref-{tag}.bin"
    arr = np.asarray(tensor.detach().cpu(), dtype=np.float32)
    arr.tofile(fn)
    print(f"Saved: {fn} {tuple(arr.shape)}")

# Register hooks for every named module except root.
hooks = []
for name, m in net.named_modules():
    if not name:
        continue
    def _make_hook(tag):
        return lambda mod, inp, out: dump(tag, out)
    hooks.append(m.register_forward_hook(_make_hook(name)))

# Dummy input.
x = torch.arange(1*12*94, dtype=torch.float32).view(1,12,94)/10
net(x)

# -Clean up hooks.
for h in hooks:
    h.remove()
print("PyTorch dump done")
