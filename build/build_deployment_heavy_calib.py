#!/usr/bin/env python3
"""Build a deployment-scene-heavy calibration subset: 250 images from the
dominant scene cluster (deployment scene) + 250 round-robin from other clusters.

This biases int8 activation calibration toward the Pi camera's distribution
while still seeing other scenes — addressing the V2-style int8 confidence
collapse seen when calibration over-represents non-deployment scenes."""
from __future__ import annotations
import json, random
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
DS = REPO / "downloadedUpdatedProductiondata"
SEED = 42
N_DOMINANT = 250
N_OTHER = 250

cj = json.loads((REPO / "build" / "v2_clusters.json").read_text())
cluster_of = cj["cluster_assignments"]
paths = [Path(p) for p in cj["image_paths"]]

cnt = Counter(cluster_of)
dom_cid, dom_size = cnt.most_common(1)[0]
print(f"dominant cluster id={dom_cid} size={dom_size}")

by_cid: dict[int, list[int]] = {}
for i, c in enumerate(cluster_of):
    by_cid.setdefault(c, []).append(i)

rng = random.Random(SEED)
# 250 from dominant cluster
dom_pool = by_cid[dom_cid][:]
rng.shuffle(dom_pool)
dom_pick = sorted(dom_pool[:N_DOMINANT])

# 250 round-robin from OTHER clusters (largest first, but exclude dominant)
other_cids = [cid for cid in by_cid if cid != dom_cid]
other_cids.sort(key=lambda cid: -len(by_cid[cid]))
other_iters: dict[int, list[int]] = {}
for cid in other_cids:
    members = by_cid[cid][:]
    rng.shuffle(members)
    other_iters[cid] = members

other_pick: list[int] = []
while len(other_pick) < N_OTHER:
    progressed = False
    for cid in other_cids:
        if other_iters[cid]:
            other_pick.append(other_iters[cid].pop(0))
            progressed = True
            if len(other_pick) >= N_OTHER:
                break
    if not progressed:
        break
other_pick = sorted(set(other_pick))
print(f"dominant pick: {len(dom_pick)} / {dom_size}")
print(f"other pick:    {len(other_pick)} / {sum(len(by_cid[c]) for c in other_cids)}")

picks = sorted(set(dom_pick) | set(other_pick))
out = DS / "calib_deployment_heavy_500.txt"
out.write_text("\n".join(str(paths[i]) for i in picks) + "\n")
print(f"wrote {out} ({len(picks)} images)")

# Also write data yaml
yml = DS / "data_calib_deployment_heavy.yaml"
yml.write_text(f"path: {DS}\ntrain: calib_deployment_heavy_500.txt\nval: calib_deployment_heavy_500.txt\n\nnames:\n  0: Target\n")
print(f"wrote {yml}")
