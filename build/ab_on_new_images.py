#!/usr/bin/env python3
"""A/B on the 240 NEW images (added in V2 batch but not present in V1 pool).
This is where V1 struggles and V3 should shine."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "build"))
from local_ab_validate import run_pt, run_tflite_f32, run_tflite_int8, summarize  # type: ignore
import json

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60, help="Sample this many of the new images (deterministic)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="build/v3_vs_v1_new_images.json")
args = ap.parse_args()

new_imgs_list = (REPO / "build" / "new_images_v2_not_v1.txt").read_text().strip().splitlines()
v2_dir = REPO / "downloadedUpdatedProductiondata" / "images"
all_new = [v2_dir / name for name in new_imgs_list]
all_new = [p for p in all_new if p.exists()]
print(f"new images on disk: {len(all_new)}")

import random
rng = random.Random(args.seed)
sample = sorted(rng.sample(all_new, min(args.n, len(all_new))))
print(f"sampling {len(sample)} NEW images")

report = {"n_sampled": len(sample), "set": "new_images_v2_not_v1"}

print("\n[V1 .pt]")
r = run_pt(REPO / "export" / "FullDataSetProd.pt", sample)
report["v1_pt"] = summarize("V1 .pt              ", r)

print("\n[V3 .pt]")
r = run_pt(REPO / "export" / "FullDataSetProdV3.pt", sample)
report["v3_pt"] = summarize("V3 .pt              ", r)

print("\n[V1 float32 TFLite]")
r = run_tflite_f32(REPO / "export/FullDataSetProd_saved_model/FullDataSetProd_float32.tflite", sample)
report["v1_f32"] = summarize("V1 float32 TFLite   ", r)

print("\n[V3 float32 TFLite]")
r = run_tflite_f32(REPO / "export/FullDataSetProdV3_saved_model/FullDataSetProdV3_float32.tflite", sample)
report["v3_f32"] = summarize("V3 float32 TFLite   ", r)

print("\n[V1 int8 (CPU)]")
r = run_tflite_int8(REPO / "export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_dwfix_v3.tflite", sample)
report["v1_int8"] = summarize("V1 int8 (CPU)       ", r)

print("\n[V3 int8 per-cluster (CPU)]")
r = run_tflite_int8(REPO / "export/FullDataSetProdV3_saved_model/FullDataSetProdV3_full_integer_quant_dwfix_v3.tflite", sample)
report["v3_int8_percluster"] = summarize("V3 int8 percluster  ", r)

print("\n[V3 int8 deployment-heavy (CPU)]")
r = run_tflite_int8(REPO / "export/FullDataSetProdV3_saved_model/FullDataSetProdV3_full_integer_quant_dwfix_depheavy_v3.tflite", sample)
report["v3_int8_depheavy"] = summarize("V3 int8 depheavy    ", r)

(REPO / args.out).write_text(json.dumps(report, indent=2))
print(f"\n[done] wrote {args.out}")
