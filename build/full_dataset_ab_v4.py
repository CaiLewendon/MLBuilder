#!/usr/bin/env python3
"""V4 full A/B: run V1 int8, V3 depheavy, V4 depheavy on ALL 4,340 V4 images.
Aggregate per-cluster + overall + new-in-V4 vs V3-era image set.
Single image load per row, all interpreters reused."""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent

def letterbox640(img: Image.Image) -> np.ndarray:
    sz = 640
    w, h = img.size
    s = min(sz / w, sz / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (sz, sz), (114, 114, 114))
    canvas.paste(img, ((sz - nw) // 2, (sz - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)

# Load V4 clusters
cj = json.loads((REPO / "build" / "v4_clusters.json").read_text())
cluster_of = cj["cluster_assignments"]
paths = [Path(p) for p in cj["image_paths"]]
print(f"images: {len(paths)}")

# Identify "new" images (in V4 pool but not V3 pool)
new_names = set((REPO / "build" / "new_images_v4_not_v3.txt").read_text().strip().splitlines())
is_new = [p.name in new_names for p in paths]
print(f"new-in-V4 (not in V3): {sum(is_new)}")

# Load int8 interpreters
import tensorflow as tf
MODELS = {
    "V1_int8":     REPO / "export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_dwfix_v3.tflite",
    "V3_depheavy": REPO / "export/FullDataSetProdV3_saved_model/FullDataSetProdV3_full_integer_quant_dwfix_depheavy_v3.tflite",
    "V4_depheavy": REPO / "export/FullDataSetProdV4_saved_model/FullDataSetProdV4_full_integer_quant_dwfix_depheavy_v3.tflite",
}
interps = {}
for name, mp in MODELS.items():
    if not mp.exists():
        raise SystemExit(f"missing model: {mp}")
    intp = tf.lite.Interpreter(model_path=str(mp))
    intp.allocate_tensors()
    in_det = intp.get_input_details()[0]
    out_det = intp.get_output_details()[0]
    interps[name] = (intp, in_det, out_det)
    print(f"loaded {name}: in_scale={in_det['quantization'][0]:.6f} in_zero={in_det['quantization'][1]}, out_scale={out_det['quantization'][0]:.6f} out_zero={out_det['quantization'][1]}")

results = {name: np.zeros(len(paths), dtype=np.float32) for name in MODELS}
t0 = time.monotonic()
for i, p in enumerate(paths):
    arr = letterbox640(Image.open(p).convert('RGB'))
    x = arr.astype(np.float32) / 255.0
    for name, (intp, in_det, out_det) in interps.items():
        in_scale, in_zero = in_det['quantization']
        out_scale, out_zero = out_det['quantization']
        qx = np.round(x / in_scale + in_zero).clip(-128, 127).astype(np.int8)
        qx = qx[np.newaxis, ...]
        intp.set_tensor(in_det['index'], qx)
        intp.invoke()
        raw = intp.get_tensor(out_det['index'])
        out = (raw.astype(np.float32) - out_zero) * out_scale
        confs = out[0, 4, :] if out.shape[1] == 5 else out[0, :, 4]
        results[name][i] = float(np.max(confs))
    if (i + 1) % 100 == 0:
        elapsed = time.monotonic() - t0
        rate = (i + 1) / elapsed
        eta = (len(paths) - i - 1) / rate
        print(f"  {i+1}/{len(paths)}  {rate:.1f} img/s  ETA {eta/60:.1f} min")

print(f"\ntotal time: {(time.monotonic()-t0)/60:.1f} min")

THRESHOLDS = [0.25, 0.5, 0.75]
def stat(confs: np.ndarray) -> dict:
    return {
        "n": int(len(confs)),
        "mean": float(confs.mean()),
        "median": float(np.median(confs)),
        "p25": float(np.percentile(confs, 25)),
        "p75": float(np.percentile(confs, 75)),
        "hit_25": int((confs >= 0.25).sum()),
        "hit_50": int((confs >= 0.50).sum()),
        "hit_75": int((confs >= 0.75).sum()),
    }

report = {"n_total": len(paths), "thresholds": THRESHOLDS, "per_model_overall": {}}

print("\n=== OVERALL (all 4,340 images) ===")
print(f"{'Model':<14} {'mean':>6} {'med':>6} {'hit>=0.25':>15} {'hit>=0.5':>15} {'hit>=0.75':>15}")
for name in MODELS:
    s = stat(results[name])
    report["per_model_overall"][name] = s
    print(f"{name:<14} {s['mean']:>6.3f} {s['median']:>6.3f}  {s['hit_25']:>4}/{s['n']:<4} ({100*s['hit_25']/s['n']:>4.1f}%)  {s['hit_50']:>4}/{s['n']:<4} ({100*s['hit_50']/s['n']:>4.1f}%)  {s['hit_75']:>4}/{s['n']:<4} ({100*s['hit_75']/s['n']:>4.1f}%)")

mask_new = np.array(is_new, dtype=bool)
print(f"\n=== ON NEW IMAGES ({int(mask_new.sum())} added in V4 batch, not in V3) ===")
print(f"{'Model':<14} {'mean':>6} {'med':>6} {'hit>=0.25':>15}")
report["per_model_new"] = {}
for name in MODELS:
    s = stat(results[name][mask_new])
    report["per_model_new"][name] = s
    print(f"{name:<14} {s['mean']:>6.3f} {s['median']:>6.3f}  {s['hit_25']:>4}/{s['n']:<4} ({100*s['hit_25']/s['n']:>4.1f}%)")

mask_old = ~mask_new
print(f"\n=== ON V3-ERA IMAGES ({int(mask_old.sum())} images) ===")
print(f"{'Model':<14} {'mean':>6} {'med':>6} {'hit>=0.25':>15}")
report["per_model_v3era"] = {}
for name in MODELS:
    s = stat(results[name][mask_old])
    report["per_model_v3era"][name] = s
    print(f"{name:<14} {s['mean']:>6.3f} {s['median']:>6.3f}  {s['hit_25']:>4}/{s['n']:<4} ({100*s['hit_25']/s['n']:>4.1f}%)")

cnt = Counter(cluster_of)
top_cids = [cid for cid, _ in cnt.most_common(10)]
print("\n=== PER-CLUSTER (top 10 clusters by size + REST) ===")
print(f"{'cid':>5} {'size':>5}  " + "  ".join(f"{name:>26}" for name in MODELS))
report["per_cluster"] = {}
for cid in top_cids:
    mask = np.array([c == cid for c in cluster_of], dtype=bool)
    size = int(mask.sum())
    parts = []
    cluster_stats = {}
    for name in MODELS:
        s = stat(results[name][mask])
        cluster_stats[name] = s
        parts.append(f"med={s['median']:.3f} hits={100*s['hit_25']/s['n']:>4.1f}%")
    report["per_cluster"][f"cid_{cid}"] = {"size": size, **cluster_stats}
    print(f"{cid:>5} {size:>5}  " + "  ".join(f"{p:>26}" for p in parts))

mask_rest = np.array([c not in top_cids for c in cluster_of], dtype=bool)
size_rest = int(mask_rest.sum())
parts = []
cluster_stats = {}
for name in MODELS:
    s = stat(results[name][mask_rest])
    cluster_stats[name] = s
    parts.append(f"med={s['median']:.3f} hits={100*s['hit_25']/s['n']:>4.1f}%")
report["per_cluster"]["REST"] = {"size": size_rest, **cluster_stats}
print(f"{'REST':>5} {size_rest:>5}  " + "  ".join(f"{p:>26}" for p in parts))

# Dominant 990 cluster headline
dom_cid = cnt.most_common(1)[0][0]
dom_size = cnt[dom_cid]
mask_dom = np.array([c == dom_cid for c in cluster_of], dtype=bool)
print(f"\n=== DOMINANT CLUSTER cid={dom_cid} size={dom_size} (deployment scene) ===")
print(f"{'Model':<14} {'mean':>6} {'med':>6} {'hit>=0.25':>15}")
report["per_model_dominant"] = {}
for name in MODELS:
    s = stat(results[name][mask_dom])
    report["per_model_dominant"][name] = s
    print(f"{name:<14} {s['mean']:>6.3f} {s['median']:>6.3f}  {s['hit_25']:>4}/{s['n']:<4} ({100*s['hit_25']/s['n']:>4.1f}%)")

(REPO / "build" / "full_dataset_ab_v4.json").write_text(json.dumps(report, indent=2))
print(f"\n[done] wrote build/full_dataset_ab_v4.json")
