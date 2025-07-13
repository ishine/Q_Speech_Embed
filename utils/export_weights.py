#!/usr/bin/env python3
# export_weights.py – dump BitGateNet weights for the C inference runtime
import os, numpy as np, torch
CKPT_PATH = "checkpoints/b_m_0.8298.pth"
OUT_DIR   = "infer_c"
DEVICE    = "cpu"
USE_QAT   = True
VERBOSE   = True

from model import BitGateNet
from utils.symquant8bit import SymQuant8bit
net  = BitGateNet(num_classes=8, quantscale=1.0, q_en=USE_QAT).to(DEVICE)
net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=False)
net.eval()
quant = SymQuant8bit()

entries, blob_lines, idx, max_sz = [], [], 0, 0

def make_entry(cname, tag, tensor, tp):
    shp = list(tensor.shape) + [1]*(4-tensor.ndim)
    return dict(cname=cname, name=tag, size=tensor.numel(),
                type=tp, dims=tensor.ndim, shape=shp[:4])

def add_blob(tag, tensor, quantise):
    global idx, max_sz
    tensor = tensor.detach().cpu()
    max_sz = max(max_sz, tensor.numel())

    if quantise:                                  # INT-8 weight + scale
        q_data, scale = quant.quantize(tensor)

        cname_w = f"w_{idx:04d}"; idx += 1
        flat_q  = q_data.view(-1).numpy().astype(np.int8)
        blob_lines.append(
            f"const int8_t  {cname_w}[] = {{ " +
            ", ".join(str(int(v)) for v in flat_q) + " };")
        entries.append(make_entry(cname_w, tag, q_data, tp=0))

        cname_s = f"w_{idx:04d}"; idx += 1
        flat_s  = scale.view(-1).cpu().numpy().astype(np.float32)
        blob_lines.append(                        #  <<< FIXED
            f"const float   {cname_s}[] = {{ " +
            ", ".join(f"{v:.8f}" for v in flat_s) + " };")
        entries.append(make_entry(
            cname_s, tag.replace("weight", "scale.param"), scale, tp=1))

        if VERBOSE:
            print(f"[INT8] {tag:40s} -> blobs #{idx-2}/{idx-1}")

    else:                                         # FP-32 blob
        cname = f"w_{idx:04d}"; idx += 1
        flat  = tensor.view(-1).numpy().astype(np.float32)
        blob_lines.append(
            f"const float   {cname}[] = {{ " +
            ", ".join(f"{v:.6f}" for v in flat) + " };")
        entries.append(make_entry(cname, tag, tensor, tp=1))
        if VERBOSE:
            print(f"[FP32] {tag:40s} -> blob  #{idx-1}")

# walk network.
for name, module in net.named_modules():
    if name == "":   continue
    local = {k:v for k,v in module.state_dict(keep_vars=True).items()
             if '.' not in k and not k.startswith(("running_", "num_batches"))}
    if not local:  continue
    for k,v in local.items():
        tag = f"{name}.{k}"
        if k=="weight": add_blob(tag,v,quantise=USE_QAT)
        elif k=="bias": add_blob(tag,v,quantise=False)
        else:          add_blob(tag,v,quantise=True)

# Emit C sources.
os.makedirs(OUT_DIR, exist_ok=True)
open(os.path.join(OUT_DIR,"weights.c"),"w").write(
    "#include <stdint.h>\n\n" + "\n".join(blob_lines) + "\n")

meta_c = [
    "#include <stdint.h>",
    '#include "weights_meta.h"', ""
]
for e in entries:
    meta_c.append(f"extern const {'int8_t' if e['type']==0 else 'float'} "
                  f"{e['cname']}[];")
meta_c.append("\nconst WT_Entry g_wt_table[] = {")
for e in entries:
    meta_c.append(
        f"  {{ {e['cname']}, {e['size']}, {e['type']}, {e['dims']}, "
        f"{{{', '.join(map(str, e['shape']))}}}, \"{e['name']}\" }},")
meta_c.append("};")
meta_c.append(f"const uint32_t g_wt_count = {len(entries)};")
open(os.path.join(OUT_DIR,"weights_meta.c"),"w").write("\n".join(meta_c))

header = [
    "#ifndef WEIGHTS_META_H",
    "#define WEIGHTS_META_H",
    "#include <stdint.h>", "",
    "typedef struct { const void *ptr; uint32_t size; uint8_t type; "
    "uint8_t dims; uint8_t shape[4]; const char *name; } WT_Entry;", "",
    f"#define TENSOR_SCRATCH_LIMIT {max_sz}", "",
    "extern const WT_Entry g_wt_table[];",
    "extern const uint32_t g_wt_count;", "",
    "#endif /* WEIGHTS_META_H */"
]
open(os.path.join(OUT_DIR,"weights_meta.h"),"w").write("\n".join(header))

print(f"\nWrote {len(entries)} blobs --> weights.c / weights_meta.c / weights_meta.h")
