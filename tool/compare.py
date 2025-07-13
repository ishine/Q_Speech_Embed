import pickle, numpy as np, glob, os

def read_c_dump(path):
    with open(path, "rb") as f:
        dims = int.from_bytes(f.read(1), 'little')
        shape_bytes = f.read(dims)
        shape = [b if isinstance(b, int) else ord(b) for b in shape_bytes]
        size  = np.prod(shape)
        data  = np.frombuffer(f.read(), dtype=np.int8)[:size]
        return np.reshape(data, shape)

ref = pickle.load(open("infer_c/acts_pt.pkl", "rb"))

for cfile in sorted(glob.glob("infer_c/dump_*.bin")):
    tag = os.path.basename(cfile)[5:-4]
    c_act = read_c_dump(cfile).astype(np.float32)
    if tag not in ref:
        print(f"{tag:12s} not found in reference!")
        continue

    pt_val = ref[tag]
    if hasattr(pt_val, "detach"):
        pt_act = pt_val.detach().cpu().numpy().astype(np.float32)
    else:
        pt_act = np.array(pt_val, dtype=np.float32)

    # Match dimensions if needed
    if pt_act.shape != c_act.shape:
        print(f"Shape mismatch for {tag}: C-shape {c_act.shape}, PT-shape {pt_act.shape}")
        # queeze batch dim if PT has 4 dims and first is 1.
        if pt_act.ndim == 4 and pt_act.shape[0] == 1:
            pt_act = pt_act[0]
        # Now check again.
        if pt_act.shape != c_act.shape:
            print(f"!! Still mismatched, skipping {tag}")
            continue

    mse = np.mean((c_act.flatten() - pt_act.flatten()) ** 2)
    print(f"{tag:12s}  mse={mse:.4g}")
