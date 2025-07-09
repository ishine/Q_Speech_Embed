from model.BitGateNet import BitGateNet
import torch, os

model_p = "models/best_model_0.8128.pth"



m  = BitGateNet(num_classes=8, quantscale=1.0)
m.load_state_dict(torch.load("models/best_model_0.8128.pth", map_location="cpu"))
print("children =", [n for n,_ in m.named_children()])          # should list conv1, block1…
print("params   =", sum(1 for _ in m.parameters()))             # should be >0
