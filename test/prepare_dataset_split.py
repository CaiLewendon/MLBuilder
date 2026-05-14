#!/usr/bin/env python3
"""
Stage a Label Studio YOLO export for Ultralytics training + EdgeTPU int8 export.

Performs a scene-aware 80/20 train/val split using perceptual-hash (dhash)
clustering so near-duplicate video frames don't leak from train into val.
Also writes the full-set calibration helper for int8 export.

Writes into <dataset>/:
    data.yaml         Ultralytics dataset descriptor (train.txt / val.txt)
    train.txt         Absolute image paths, ~80% of dataset, scene-grouped
    val.txt           Absolute image paths, ~20% of dataset, scene-grouped
    calib_all.txt     All image paths, for int8 calibration coverage
    data_calib.yaml   Calibration-only descriptor (train=val=calib_all.txt)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(img_path: Path, hash_size: int = 8) -> int:
    img = Image.open(img_path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_by_dhash(hashes: list[int], threshold: int) -> list[list[int]]:
    n = len(hashes)
    uf = UnionFind(n)
    for i in range(n):
        hi = hashes[i]
        for j in range(i + 1, n):
            if (hi ^ hashes[j]).bit_count() <= threshold:
                uf.union(i, j)
    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(uf.find(i), []).append(i)
    return list(buckets.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="project-1-at-2026-05-13-06-40-8e81e090",
        help="Dataset folder name (relative to repo root) or absolute path.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=5,
        help="Hamming distance <= threshold counts as same scene. Lower = stricter (fewer chained clusters).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ds_arg = Path(args.dataset)
    ds = ds_arg if ds_arg.is_absolute() else (repo_root / ds_arg)
    ds = ds.resolve()
    img_dir = ds / "images"
    label_dir = ds / "labels"
    if not img_dir.is_dir():
        sys.exit(f"missing images dir: {img_dir}")
    if not label_dir.is_dir():
        sys.exit(f"missing labels dir: {label_dir}")

    paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    n = len(paths)
    if n == 0:
        sys.exit("no images found")

    missing_labels = [p.name for p in paths if not (label_dir / (p.stem + ".txt")).is_file()]
    if missing_labels:
        print(
            f"[warn] {len(missing_labels)} images have no matching label file "
            f"(YOLO treats these as negatives). First 3: {missing_labels[:3]}"
        )

    print(f"[hash] {n} images -> computing dhash")
    hashes: list[int] = []
    for i, p in enumerate(paths):
        hashes.append(dhash(p))
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}")

    print(f"[cluster] threshold={args.hash_threshold}")
    clusters = cluster_by_dhash(hashes, args.hash_threshold)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(
        f"[cluster] {len(clusters)} clusters | "
        f"largest={sizes[0]} median={sizes[len(sizes)//2]} singletons={sum(1 for s in sizes if s == 1)}"
    )

    target_val = int(round(n * args.val_ratio))
    target_train = n - target_val
    rng = random.Random(args.seed)
    # Largest-first balanced bin packing: each cluster goes to whichever side
    # is furthest below its quota (with random tiebreak). Ties at quota = 0%.
    clusters_sorted = sorted(clusters, key=lambda c: (-len(c), rng.random()))
    val_set: set[int] = set()
    train_count = 0
    for cluster in clusters_sorted:
        val_deficit = target_val - len(val_set)
        train_deficit = target_train - train_count
        # Normalize by remaining capacity so a near-full bucket doesn't keep grabbing clusters.
        val_pressure = val_deficit / target_val if target_val > 0 else 0
        train_pressure = train_deficit / target_train if target_train > 0 else 0
        if val_pressure > train_pressure:
            for i in cluster:
                val_set.add(i)
        else:
            train_count += len(cluster)

    train_paths = [paths[i] for i in range(n) if i not in val_set]
    val_paths = [paths[i] for i in sorted(val_set)]
    actual_ratio = len(val_paths) / n
    print(
        f"[split] train={len(train_paths)} val={len(val_paths)} "
        f"(target val={target_val}, actual ratio={actual_ratio:.3f})"
    )

    def write_list(path: Path, items: list[Path]) -> None:
        path.write_text("\n".join(str(p) for p in items) + "\n")
        print(f"[write] {path.relative_to(repo_root)} ({len(items)} entries)")

    write_list(ds / "train.txt", train_paths)
    write_list(ds / "val.txt", val_paths)
    write_list(ds / "calib_all.txt", paths)

    data_yaml = ds / "data.yaml"
    data_yaml.write_text(
        f"path: {ds}\n"
        f"train: train.txt\n"
        f"val: val.txt\n"
        f"\n"
        f"names:\n"
        f"  0: Target\n"
    )
    print(f"[write] {data_yaml.relative_to(repo_root)}")

    data_calib = ds / "data_calib.yaml"
    data_calib.write_text(
        f"path: {ds}\n"
        f"train: calib_all.txt\n"
        f"val: calib_all.txt\n"
        f"\n"
        f"names:\n"
        f"  0: Target\n"
    )
    print(f"[write] {data_calib.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
