#!/usr/bin/env python3
"""Run dhash clustering on V2's image pool and print size distribution."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "test"))
from prepare_dataset_split import dhash, cluster_by_dhash  # type: ignore

DATASET = REPO / "downloadedUpdatedProductiondata"
HASH_THRESHOLD = 5

img_dir = DATASET / "images"
paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
print(f"images: {len(paths)}")

print("computing dhash...")
hashes = []
for i, p in enumerate(paths):
    hashes.append(dhash(p))
    if (i + 1) % 500 == 0 or i + 1 == len(paths):
        print(f"  {i + 1}/{len(paths)}")

clusters = cluster_by_dhash(hashes, HASH_THRESHOLD)
sizes = sorted((len(c) for c in clusters), reverse=True)
print(f"\nclusters: {len(clusters)}")
print(f"  largest 20 sizes: {sizes[:20]}")
print(f"  median size: {sizes[len(sizes)//2]}")
print(f"  singletons: {sum(1 for s in sizes if s == 1)}")
print(f"  >100 imgs: {sum(1 for s in sizes if s > 100)}")
print(f"  >50 imgs: {sum(1 for s in sizes if s > 50)}")
print(f"  >20 imgs: {sum(1 for s in sizes if s > 20)}")
print(f"  >10 imgs: {sum(1 for s in sizes if s > 10)}")

print("\ncluster size buckets:")
buckets = [(1,1),(2,5),(6,10),(11,20),(21,50),(51,100),(101,200),(201,500),(501,1000),(1001,9999)]
for lo, hi in buckets:
    n = sum(1 for s in sizes if lo <= s <= hi)
    imgs = sum(s for s in sizes if lo <= s <= hi)
    print(f"  size {lo:>4}-{hi:>4}: {n:>4} clusters, {imgs:>5} images")

# save cluster-to-image mapping for the split script
import json
out = {
    "n_images": len(paths),
    "hash_threshold": HASH_THRESHOLD,
    "image_paths": [str(p) for p in paths],
    "cluster_assignments": [],
}
# For each image (by index in paths), determine its cluster id
img_to_cluster = {}
for cid, members in enumerate(clusters):
    for idx in members:
        img_to_cluster[idx] = cid
out["cluster_assignments"] = [img_to_cluster[i] for i in range(len(paths))]
out["cluster_sizes"] = sizes
(REPO / "build" / "v2_clusters.json").write_text(json.dumps(out))
print(f"\nwrote build/v2_clusters.json ({len(paths)} images, {len(clusters)} clusters)")
