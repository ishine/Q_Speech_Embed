# tiny_speech/bin_gen.py
"""
Export one validation sample + every intermediate activation
in a binary format that tiny_c/* tests read.

The first convolution layer (3×3, stride=1, pad=1, bias-free) is recomputed
with *exactly* the same math used in tiny_c/conv2d.c:
    • INT8 weights   (single scale value for the whole tensor)
    • per-channel fake-quant of activations
That guarantees a byte-for-byte match between Python and C.
"""
import os, yaml, torch, numpy as np
from pathlib import Path
import torch.nn.functional as F                     # ★ NEW

from model.BitGateNet     import BitGateNet
from dataset.SCDataset      import SCDataset
from Preprocess     import preprocess_audio_batch
from utils.symquant8bit   import SymQuant8bit              # identical to tiny-C

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
def save_tensor(t: torch.Tensor, fname: str) -> None:
    """Write <shape as float32>  +  <FP32 flattened data>."""
    t = t.detach().cpu()
    if t.ndim and t.shape[0] == 1:          # drop batch-dim if size-1
        t = t.squeeze(0)

    shape = np.asarray(t.shape,  dtype=np.float32)
    data  = t.flatten().numpy().astype(np.float32)

    with open(LOG_DIR / fname, "wb") as f:
        f.write(shape.tobytes())
        f.write(data.tobytes())

    print(f"✓ {fname:<20}  shape={tuple(shape.astype(int))}  head={data[:5]}")

# ──────────────────────────────────────────────────────────────
def fake_quant_per_channel(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric per-channel fake-quant to INT8 and back to FP32.
    x: (1, C_in, H, W) → same shape fp32
    """
    amax  = x.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    scale = 127.0 / amax
    x_q   = (x * scale).round().clamp(-127, 127)
    return x_q / scale

# ──────────────────────────────────────────────────────────────
def conv2d_int8_like_c(x, weight_fp, quantizer,
                       stride=1, pad=1, bias=None):
    """
    Pure-Python re-implementation of tiny_c/conv2d.c (slow, reference only).
    x         : (1, C_in, H, W)  FP32 activations
    weight_fp : (C_out, C_in, 3, 3) FP32 weights (we quantise here)
    """
    _, C_in, H_in, W_in       = x.shape
    C_out, _, K_h, K_w        = weight_fp.shape
    H_out = (H_in + 2*pad - K_h)//stride + 1
    W_out = (W_in + 2*pad - K_w)//stride + 1

    # -------- INT8 weights + scale (same quantiser used in C) --------
    w_q, w_scale = quantizer.quantize(weight_fp)   # single scale value here
    w_q     = w_q.to(torch.int8)
    w_scale = w_scale.item()

    # -------- fake-quantise activations per channel -----------------
    x_fq = fake_quant_per_channel(x)

    # -------- pad once so every 3×3 slice exists -------------------- ★
    x_pad = F.pad(x_fq, (pad, pad, pad, pad))      # (1,C,H+2p,W+2p)

    out = torch.zeros((1, C_out, H_out, W_out), dtype=torch.float32)

    for oc in range(C_out):
        w_oc = (w_q[oc].to(torch.float32) / w_scale)   # (C_in,3,3)

        for oh in range(H_out):
            ih0 = oh * stride
            for ow in range(W_out):
                iw0 = ow * stride
                acc = 0.0 if bias is None else float(bias[oc])

                for ic in range(C_in):
                    xs = x_pad[0, ic, ih0:ih0+K_h, iw0:iw0+K_w]   # 3×3 slice
                    acc += torch.sum(xs * w_oc[ic]).item()

                out[0, oc, oh, ow] = acc

    return out

# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    # ---------- load config & validation sample ---------------------
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    labels  = cfg["labels"]
    val_ds  = SCDataset(cfg["dataset"]["val_path"], labels)
    val_ld  = torch.utils.data.DataLoader(
                val_ds, batch_size=1, shuffle=False,
                collate_fn=preprocess_audio_batch)

    inputs, targets = next(iter(val_ld))
    print(f"sample label: {labels[targets.item()]}")

    save_tensor(inputs, "00_input.bin")

    # ---------- load TinySpeech checkpoint -------------------------
    ckpt = sorted(Path("models").glob("*.pth"))[-1]
    model = BitGateNet(len(labels), quantscale=cfg["quant_scale"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    quantizer = model.conv1.quantizer

    print(f"ckpt loaded: {ckpt.name}")

    # ---------- Conv-1 (exactly the C math) ------------------------
    conv1_out = conv2d_int8_like_c(inputs,
                                   model.conv1.weight.detach(),
                                   quantizer, stride=1, pad=1, bias=None)
    conv1_out = torch.relu(conv1_out)
    save_tensor(conv1_out, "01_conv1_out.bin")
    # --- NEW: save the 2×-downsampled tensor for C verification ---
    pooled = torch.nn.functional.max_pool2d(conv1_out, kernel_size=2, stride=2)
    save_tensor(pooled, "01_conv1_out_pool2.bin")


    # ---------- rest of the network --------------------------------
    x = conv1_out
    x = model.block1(x);   save_tensor(x, "02_block1_out.bin")
    x = model.block2(x);   save_tensor(x, "03_block2_out.bin")
    x = model.block3(x);   save_tensor(x, "04_block3_out.bin")
    x = model.block4(x);   save_tensor(x, "05_block4_out.bin")
    x = torch.relu(model.conv2(x));    save_tensor(x, "06_conv2_out.bin")
    x = model.global_pool(x);          save_tensor(x, "07_global_pool.bin")
    x = x.view(x.size(0), -1);         save_tensor(x, "08_flattened.bin")
    x = model.fc(x);                   save_tensor(x, "09_fc_out.bin")
    save_tensor(x, "final_logits.bin")

    print("\nAll layer activations exported – ready for C verification.")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()



def log_tensor(name, tensor, outdir="logs"):
    tensor = tensor.detach().cpu().numpy()
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/{name}.bin", 'wb') as f:
        for dim in tensor.shape:
            f.write(np.array(dim, dtype=np.float32).tobytes())
        f.write(tensor.flatten().astype(np.float32).tobytes())
