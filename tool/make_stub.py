# make_stub.py 
import torch
import numpy as np

# our network's input (batch, channels, height, width).
x = torch.randn(1, 12, 94)
np.save("infer_c/stub_inp.npy", x.numpy())
print("Saved test input to infer_c/stub_inp.npy")
