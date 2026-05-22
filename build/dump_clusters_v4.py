#!/usr/bin/env python3
"""One-shot: dhash + cluster the V4 dataset and dump build/v4_clusters.json
in the format build_deployment_heavy_calib.py expects.

Output schema:
    {
      "n_images": int,
      "hash_threshold": int,
      "cluster_assignments": [int, ...],   # cluster id per image
      "image_paths": [str, ...],           # absolute paths, same order
    }
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "test"))
from prepare_dataset_split import dhash, cluster_by_dhash  # type: ignore

DATASET = REPO / "project-1-at-2026-05-22-04-22-5cace4ec"
OUT = REPO / "build" / "v4_clusters.json"
HASH_THRESHOLD = 5

img_dir = DATASET / "images"
paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
n = len(paths)
print(f"[scan] {n} images")

print(f"[hash] computing dhash for {n} images...")
hashes = []
for i, p in enumerate(paths):
    hashes.append(dhash(p))
    if (i + 1) % 500 == 0 or i + 1 == n:
        print(f"  {i+1}/{n}")

print(f"[cluster] threshold={HASH_THRESHOLD}")
clusters = cluster_by_dhash(hashes, HASH_THRESHOLD)
cluster_of = [-1] * n
for cid, members in enumerate(clusters):
    for idx in members:
        cluster_of[idx] = cid

assert all(c >= 0 for c in cluster_of)

payload = {
    "n_images": n,
    "hash_threshold": HASH_THRESHOLD,
    "cluster_assignments": cluster_of,
    "image_paths": [str(p) for p in paths],
}
OUT.write_text(json.dumps(payload))
print(f"[write] {OUT} ({n} images, {len(clusters)} clusters)")
