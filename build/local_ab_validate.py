#!/usr/bin/env python3
"""Local A/B validation: run V3 vs V1 on the deployment scene cluster.

Picks images from the 990-image dominant cluster (which IS the deployment scene
on the Pi camera) and runs each model in float32 mode + int8 mode locally.
Compares per-image confidence and detection rate. V3 must match or beat V1.

Usage:
  venv/bin/python build/local_ab_validate.py \
      --v1-weights export/FullDataSetProd.pt \
      --v3-weights export/FullDataSetProdV3.pt \
      [--v1-tflite export/FullDataSetProd_saved_model/FullDataSetProd_float32.tflite]
      [--v3-tflite export/FullDataSetProdV3_saved_model/FullDataSetProdV3_float32.tflite]
      [--v1-int8 export/FullDataSetProd_full_integer_quant_edgetpu.tflite]
      [--v3-int8 export/FullDataSetProdV3_edgetpu.tflite]
      [--n 30]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
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
    return np.asarray(canvas, dtype=np.uint8)  # H,W,3 uint8


def run_pt(weights: Path, images: list[Path]) -> list[dict]:
    """Run a .pt model via Ultralytics, return per-image best-detection results."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    results = []
    for p in images:
        r = m.predict(source=str(p), imgsz=640, conf=0.01, verbose=False)[0]
        if len(r.boxes) == 0:
            results.append({"img": p.name, "conf": 0.0, "bbox": None, "n_det": 0})
            continue
        confs = r.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        xyxy = r.boxes.xyxy.cpu().numpy()[best]
        results.append({
            "img": p.name,
            "conf": float(confs[best]),
            "bbox": [float(x) for x in xyxy],
            "n_det": int(len(r.boxes)),
        })
    return results


def run_tflite_f32(tflite_path: Path, images: list[Path]) -> list[dict]:
    """Run a float32 TFLite model on letterboxed-640 images, return best-conf detection."""
    import tensorflow as tf
    intp = tf.lite.Interpreter(model_path=str(tflite_path))
    intp.allocate_tensors()
    inp_det = intp.get_input_details()[0]
    out_det = intp.get_output_details()[0]
    in_shape = inp_det['shape']  # (1,640,640,3) or (1,3,640,640) depending on emitter
    nchw = (in_shape[1] == 3)
    results = []
    for p in images:
        arr = letterbox640(Image.open(p).convert('RGB'))  # H,W,3 uint8
        x = arr.astype(np.float32) / 255.0
        if nchw:
            x = np.transpose(x, (2, 0, 1))
        x = x[np.newaxis, ...]
        intp.set_tensor(inp_det['index'], x)
        intp.invoke()
        out = intp.get_tensor(out_det['index'])  # (1, 5, 8400) or (1, 8400, 5) etc.
        # raw head: rows are [cx, cy, w, h, class_conf]; output may be (1,5,N) or (1,N,5)
        if out.shape[1] == 5 and out.shape[2] > 5:
            confs = out[0, 4, :]
            xywh = out[0, :4, :].T
        else:
            confs = out[0, :, 4]
            xywh = out[0, :, :4]
        best = int(np.argmax(confs))
        c = float(confs[best])
        cx, cy, w, h = [float(v) for v in xywh[best]]
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        results.append({"img": p.name, "conf": c, "bbox": [x1, y1, x2, y2], "n_det": int((confs > 0.25).sum())})
    return results


def run_tflite_int8(tflite_path: Path, images: list[Path]) -> list[dict]:
    """Run an int8 EdgeTPU/quant TFLite model on CPU (no TPU available locally).
    Uses tflite_runtime if available, else tensorflow."""
    try:
        from tflite_runtime.interpreter import Interpreter
        intp = Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf
        intp = tf.lite.Interpreter(model_path=str(tflite_path))
    intp.allocate_tensors()
    inp_det = intp.get_input_details()[0]
    out_det = intp.get_output_details()[0]
    in_shape = inp_det['shape']
    nchw = (in_shape[1] == 3)
    in_scale, in_zero = inp_det['quantization']
    out_scale, out_zero = out_det['quantization']
    results = []
    for p in images:
        arr = letterbox640(Image.open(p).convert('RGB'))
        x = arr.astype(np.float32) / 255.0
        if nchw:
            x = np.transpose(x, (2, 0, 1))
        x = x[np.newaxis, ...]
        if in_scale > 0:
            qx = np.round(x / in_scale + in_zero).clip(-128, 127).astype(np.int8)
        else:
            qx = x.astype(np.int8)
        intp.set_tensor(inp_det['index'], qx)
        intp.invoke()
        raw = intp.get_tensor(out_det['index'])
        if out_scale > 0:
            out = (raw.astype(np.float32) - out_zero) * out_scale
        else:
            out = raw.astype(np.float32)
        if out.shape[1] == 5 and out.shape[2] > 5:
            confs = out[0, 4, :]
            xywh = out[0, :4, :].T
        else:
            confs = out[0, :, 4]
            xywh = out[0, :, :4]
        best = int(np.argmax(confs))
        c = float(confs[best])
        cx, cy, w, h = [float(v) for v in xywh[best]]
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        results.append({"img": p.name, "conf": c, "bbox": [x1, y1, x2, y2], "n_det": int((confs > 0.25).sum())})
    return results


def summarize(label: str, results: list[dict]) -> dict:
    confs = np.array([r["conf"] for r in results])
    n_hit = int((confs >= 0.25).sum())
    print(f"  {label:25s} n={len(results)} hits>=0.25: {n_hit}/{len(results)} ({100*n_hit/len(results):.1f}%) "
          f"mean_conf={confs.mean():.3f} median={np.median(confs):.3f} min={confs.min():.3f} max={confs.max():.3f}")
    return {"label": label, "n_total": len(results), "n_hits_25": n_hit,
            "mean_conf": float(confs.mean()), "median_conf": float(np.median(confs)),
            "min_conf": float(confs.min()), "max_conf": float(confs.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-weights", default="export/FullDataSetProd.pt")
    ap.add_argument("--v3-weights", default="export/FullDataSetProdV3.pt")
    ap.add_argument("--v1-tflite-f32", default="export/FullDataSetProd_saved_model/FullDataSetProd_float32.tflite")
    ap.add_argument("--v3-tflite-f32", default="")
    ap.add_argument("--v1-int8", default="export/FullDataSetProd_full_integer_quant_edgetpu.tflite")
    ap.add_argument("--v3-int8", default="")
    ap.add_argument("--cluster-json", default="build/v2_clusters.json")
    ap.add_argument("--n", type=int, default=30, help="Images per scene cluster to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="build/v3_vs_v1_local_ab.json")
    args = ap.parse_args()

    cj = json.loads(Path(args.cluster_json).read_text())
    cluster_of = cj["cluster_assignments"]
    paths = [Path(p) for p in cj["image_paths"]]
    # find the dominant cluster id (largest size)
    from collections import Counter
    cnt = Counter(cluster_of)
    dom_cid, dom_size = cnt.most_common(1)[0]
    print(f"[validate] dominant cluster id={dom_cid} size={dom_size} (deployment scene)")

    import random
    rng = random.Random(args.seed)
    dom_indices = [i for i, c in enumerate(cluster_of) if c == dom_cid]
    sample = sorted(rng.sample(dom_indices, min(args.n, len(dom_indices))))
    sample_paths = [paths[i] for i in sample]
    print(f"[validate] sampling {len(sample_paths)} images from cluster {dom_cid}")

    report = {"dominant_cluster": dom_cid, "dominant_size": dom_size, "n_sampled": len(sample_paths)}

    def is_real_pt(p: str) -> bool:
        return p and p != "/dev/null" and Path(p).is_file() and Path(p).suffix == ".pt"

    if is_real_pt(args.v1_weights):
        print("\n[V1 .pt]")
        r_v1_pt = run_pt(Path(args.v1_weights), sample_paths)
        report["v1_pt_summary"] = summarize("V1 .pt              ", r_v1_pt)
        report["v1_pt_per_image"] = r_v1_pt

    if is_real_pt(args.v3_weights):
        print("\n[V3 .pt]")
        r_v3_pt = run_pt(Path(args.v3_weights), sample_paths)
        report["v3_pt_summary"] = summarize("V3 .pt              ", r_v3_pt)
        report["v3_pt_per_image"] = r_v3_pt
    else:
        print(f"[V3 .pt] {args.v3_weights} not present — skip")

    if args.v1_tflite_f32 and Path(args.v1_tflite_f32).exists():
        print("\n[V1 float32 TFLite]")
        r_v1_f32 = run_tflite_f32(Path(args.v1_tflite_f32), sample_paths)
        report["v1_f32_summary"] = summarize("V1 float32 TFLite   ", r_v1_f32)
        report["v1_f32_per_image"] = r_v1_f32

    if args.v3_tflite_f32 and Path(args.v3_tflite_f32).exists():
        print("\n[V3 float32 TFLite]")
        r_v3_f32 = run_tflite_f32(Path(args.v3_tflite_f32), sample_paths)
        report["v3_f32_summary"] = summarize("V3 float32 TFLite   ", r_v3_f32)
        report["v3_f32_per_image"] = r_v3_f32

    if args.v1_int8 and Path(args.v1_int8).exists():
        print("\n[V1 int8 EdgeTPU TFLite (CPU)]")
        r_v1_i8 = run_tflite_int8(Path(args.v1_int8), sample_paths)
        report["v1_int8_summary"] = summarize("V1 int8 (CPU)       ", r_v1_i8)
        report["v1_int8_per_image"] = r_v1_i8

    if args.v3_int8 and Path(args.v3_int8).exists():
        print("\n[V3 int8 EdgeTPU TFLite (CPU)]")
        r_v3_i8 = run_tflite_int8(Path(args.v3_int8), sample_paths)
        report["v3_int8_summary"] = summarize("V3 int8 (CPU)       ", r_v3_i8)
        report["v3_int8_per_image"] = r_v3_i8

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
