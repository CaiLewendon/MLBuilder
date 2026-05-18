#!/usr/bin/env python3
"""Robust per-cluster-capped train/val split for maximum scene diversity.

For each dhash scene cluster of size S:
  val_n  = min(round(S*0.2), CAP_VAL) if S >= 2 else 0
  train_n = min(S - val_n, CAP_TRAIN)
  holdout_n = S - val_n - train_n

CAP_TRAIN = 100  caps the dominant 990-image scene to 100 training images,
                 putting it on equal footing with the next 4 large clusters
                 (240/120/110/100). Smaller clusters keep all their members.

Outputs (in dataset dir):
  train.txt, val.txt, calib_all.txt, holdout.txt, data.yaml, data_calib.yaml
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "test"))
from prepare_dataset_split import dhash, cluster_by_dhash  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="downloadedUpdatedProductiondata")
    p.add_argument("--cap-train", type=int, default=100)
    p.add_argument("--cap-val", type=int, default=40)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--hash-threshold", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cluster-cache", default="build/v2_clusters.json")
    args = p.parse_args()

    ds = (REPO / args.dataset).resolve()
    img_dir = ds / "images"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n = len(paths)
    print(f"[scan] {n} images in {ds}")

    cache_path = REPO / args.cluster_cache
    if cache_path.exists():
        print(f"[cache] loading clusters from {cache_path}")
        c = json.loads(cache_path.read_text())
        assert c["n_images"] == n, f"cache stale: {c['n_images']} vs {n}"
        assert c["hash_threshold"] == args.hash_threshold
        cluster_of = c["cluster_assignments"]
    else:
        print(f"[hash] computing dhash for {n} images...")
        hashes = []
        for i, pp in enumerate(paths):
            hashes.append(dhash(pp))
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n}")
        print(f"[cluster] threshold={args.hash_threshold}")
        clusters = cluster_by_dhash(hashes, args.hash_threshold)
        cluster_of = [-1] * n
        for cid, members in enumerate(clusters):
            for idx in members:
                cluster_of[idx] = cid

    # group images by cluster
    by_cluster: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_of):
        by_cluster.setdefault(cid, []).append(idx)
    sizes = sorted((len(v) for v in by_cluster.values()), reverse=True)
    print(f"[clusters] {len(by_cluster)} clusters | largest={sizes[0]} median={sizes[len(sizes)//2]}")

    rng = random.Random(args.seed)
    train_idx, val_idx, hold_idx = [], [], []
    for cid, members in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        S = len(members)
        if S >= 2:
            val_n = min(round(S * args.val_ratio), args.cap_val)
        else:
            val_n = 0
        train_n = min(S - val_n, args.cap_train)
        hold_n = S - val_n - train_n
        shuffled = members[:]
        rng.shuffle(shuffled)
        val_idx.extend(shuffled[:val_n])
        train_idx.extend(shuffled[val_n:val_n + train_n])
        hold_idx.extend(shuffled[val_n + train_n:])

    train_idx.sort(); val_idx.sort(); hold_idx.sort()
    print(f"[split] train={len(train_idx)} val={len(val_idx)} holdout={len(hold_idx)} (total={len(train_idx)+len(val_idx)+len(hold_idx)})")
    print(f"  train ratio: {len(train_idx)/n:.3f}, val ratio: {len(val_idx)/n:.3f}, holdout ratio: {len(hold_idx)/n:.3f}")

    # Calibration set: per-cluster sample, max 1-3 per cluster, target ~500
    # Round-robin pick from large clusters first
    calib_idx: list[int] = []
    target_calib = 500
    cluster_iters = {cid: iter(members) for cid, members in sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))}
    while len(calib_idx) < target_calib:
        added_any = False
        for cid in list(cluster_iters):
            it = cluster_iters[cid]
            try:
                idx = next(it)
                calib_idx.append(idx)
                added_any = True
                if len(calib_idx) >= target_calib:
                    break
            except StopIteration:
                del cluster_iters[cid]
        if not added_any:
            break
    calib_idx = sorted(set(calib_idx))
    print(f"[calib] {len(calib_idx)} images (target {target_calib}, per-cluster round-robin)")

    def write_list(fname: str, indices: list[int]):
        out = ds / fname
        out.write_text("\n".join(str(paths[i]) for i in indices) + "\n")
        print(f"[write] {out.relative_to(REPO)} ({len(indices)})")

    write_list("train.txt", train_idx)
    write_list("val.txt", val_idx)
    write_list("holdout.txt", hold_idx)
    write_list("calib_all.txt", calib_idx)

    (ds / "data.yaml").write_text(
        f"path: {ds}\ntrain: train.txt\nval: val.txt\n\nnames:\n  0: Target\n"
    )
    print(f"[write] data.yaml")

    (ds / "data_calib.yaml").write_text(
        f"path: {ds}\ntrain: calib_all.txt\nval: calib_all.txt\n\nnames:\n  0: Target\n"
    )
    print(f"[write] data_calib.yaml")

    # write a summary
    summary = {
        "cap_train": args.cap_train, "cap_val": args.cap_val,
        "n_total": n, "n_train": len(train_idx), "n_val": len(val_idx), "n_holdout": len(hold_idx), "n_calib": len(calib_idx),
        "cluster_count": len(by_cluster), "largest_cluster": sizes[0],
    }
    (ds / "robust_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
