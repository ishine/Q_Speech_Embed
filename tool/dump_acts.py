# tool/dump_acts.py
import torch
import pickle
import numpy as np
from model import BitGateNet 

# 1. Load.
model = BitGateNet(num_classes=8, quantscale=1.0, q_en=False)
ckpt = torch.load("checkpoints/b_m_0.8298.pth", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()

# 2. Prepare dummy input (match shape to what our model expects!)
x = torch.from_numpy(np.load("infer_c/stub_inp.npy")).unsqueeze(0).float() 

# 3. Hook activations.
acts = {}

def save_hook(name):
    def hook(module, inp, outp):
        acts[name] = outp.detach().cpu().numpy()
    return hook

# Register hooks on relevant layers
for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList) and name != "":
        module.register_forward_hook(save_hook(name))

# 4. Run forward pass
with torch.no_grad():
    model(x)

# 5. Save activations.
with open("infer_c/acts_pt.pkl", "wb") as f:
    pickle.dump(acts, f)
print(f"Saved {len(acts)} activations to infer_c/acts_pt.pkl")
