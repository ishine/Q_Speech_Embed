import numpy as np

with open("logs/01_conv1_out.bin", "rb") as f:
    shape = np.frombuffer(f.read(16), dtype=np.float32)
    print("Shape header (as float32):", shape)

    data = np.frombuffer(f.read(), dtype=np.float32)
    print("Data length:", data.size)