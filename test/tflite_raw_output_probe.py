#!/usr/bin/env python3
"""
Standalone TFLite raw-output diagnostic script.

Purpose:
- Load a model directly with tflite_runtime (no custom wrappers)
- Print full input/output tensor metadata
- Run inference on black/white/random inputs
- Dump raw output tensors + min/max/mean stats
- Provide simple heuristics about output plausibility and quantization sanity
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "target_detector_int8_edgetpu.tflite"
DEFAULT_TPU_LIB = "libedgetpu.so.1"


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def print_tensor_detail_block(title: str, details: list[dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    for d in details:
        print("-" * 80)
        print(f"name: {d.get('name')}")
        print(f"index: {d.get('index')}")
        print(f"shape: {d.get('shape')}")
        print(f"shape_signature: {d.get('shape_signature')}")
        print(f"dtype: {d.get('dtype')}")
        print(f"quantization: {d.get('quantization')}")
        qparams = d.get("quantization_parameters", {})
        print("quantization_parameters:")
        print(f"  scales: {qparams.get('scales')}")
        print(f"  zero_points: {qparams.get('zero_points')}")
        print(f"  quantized_dimension: {qparams.get('quantized_dimension')}")
        if "sparsity_parameters" in d:
            print(f"sparsity_parameters: {d.get('sparsity_parameters')}")


def quant_sanity_for_detail(detail: dict[str, Any], io_name: str) -> list[str]:
    notes: list[str] = []
    dtype = detail["dtype"]
    qparams = detail.get("quantization_parameters", {})
    scales = np.array(qparams.get("scales", []), dtype=np.float64)
    zeros = np.array(qparams.get("zero_points", []), dtype=np.int64)

    is_quant_dtype = dtype in (np.int8, np.uint8, np.int16, np.uint16)
    if not is_quant_dtype:
        notes.append(f"{io_name}: dtype={dtype} (not integer-quantized tensor)")
        return notes

    if scales.size == 0:
        notes.append(f"{io_name}: WARNING quantized dtype but scales are empty")
        return notes

    if np.any(~np.isfinite(scales)):
        notes.append(f"{io_name}: WARNING non-finite quantization scales detected")
    if np.any(scales <= 0):
        notes.append(f"{io_name}: WARNING non-positive quantization scales detected")

    if dtype == np.int8:
        if zeros.size > 0 and (np.any(zeros < -128) or np.any(zeros > 127)):
            notes.append(f"{io_name}: WARNING int8 zero-point out of range [-128,127]")
    if dtype == np.uint8:
        if zeros.size > 0 and (np.any(zeros < 0) or np.any(zeros > 255)):
            notes.append(f"{io_name}: WARNING uint8 zero-point out of range [0,255]")

    scale_min = float(np.min(scales))
    scale_max = float(np.max(scales))
    notes.append(f"{io_name}: quant scales range [{scale_min:.8g}, {scale_max:.8g}]")

    # Heuristic: int8 image inputs for normalized models often have ~1/255 scale.
    if is_quant_dtype and "input" in io_name.lower() and scale_max < 0.02:
        notes.append(
            f"{io_name}: scale suggests normalized real input range (likely 0..1 before quantization)"
        )
    elif is_quant_dtype and "input" in io_name.lower() and scale_min > 0.5:
        notes.append(
            f"{io_name}: large input scale suggests non-normalized real range (possibly 0..255-like)"
        )

    return notes


def make_synthetic_inputs(h: int, w: int, c: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    black = np.zeros((h, w, c), dtype=np.uint8)
    white = np.full((h, w, c), 255, dtype=np.uint8)
    noise = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
    return {"black": black, "white": white, "noise": noise}


def convert_image_for_input(
    image_u8: np.ndarray,
    input_detail: dict[str, Any],
    input_real_space: str = "normalized",
) -> np.ndarray:
    dtype = input_detail["dtype"]
    shape = input_detail["shape"]
    if len(shape) != 4:
        raise RuntimeError(f"Expected 4D input tensor, got shape={shape}")
    b, h, w, c = [int(x) for x in shape]
    if b != 1:
        raise RuntimeError(f"Expected batch size 1 input tensor, got batch={b}")

    if image_u8.shape != (h, w, c):
        raise RuntimeError(
            f"Synthetic image shape mismatch: expected {(h, w, c)}, got {image_u8.shape}"
        )

    # Real-space pixels used for quantization step.
    if input_real_space == "normalized":
        real = image_u8.astype(np.float32) / 255.0
    elif input_real_space == "pixels":
        real = image_u8.astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported input_real_space='{input_real_space}'")

    q_scale, q_zero = input_detail.get("quantization", (0.0, 0))
    if dtype in (np.float32, np.float16):
        tensor = real.astype(dtype)
    elif dtype in (np.int8, np.uint8, np.int16, np.uint16):
        if q_scale is None or float(q_scale) <= 0.0:
            raise RuntimeError(
                f"Quantized input dtype={dtype} but quantization scale is invalid: {q_scale}"
            )
        q = np.round(real / float(q_scale) + float(q_zero))
        info = np.iinfo(dtype)
        q = np.clip(q, info.min, info.max)
        tensor = q.astype(dtype)
    else:
        raise RuntimeError(f"Unsupported input dtype: {dtype}")

    return np.expand_dims(tensor, axis=0)


def tensor_stats(arr: np.ndarray) -> dict[str, Any]:
    arr64 = arr.astype(np.float64, copy=False)
    total = int(arr.size)
    nz = int(np.count_nonzero(arr))
    finite_ratio = float(np.isfinite(arr64).mean()) if total else 1.0
    unique_preview = None
    if total <= 50000:
        unique_preview = int(np.unique(arr).size)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": total,
        "min": float(np.min(arr64)) if total else 0.0,
        "max": float(np.max(arr64)) if total else 0.0,
        "mean": float(np.mean(arr64)) if total else 0.0,
        "std": float(np.std(arr64)) if total else 0.0,
        "nonzero_ratio": float(nz / total) if total else 0.0,
        "finite_ratio": finite_ratio,
        "unique_count_if_small": unique_preview,
    }


def classify_detection_like_shape(shape: tuple[int, ...] | list[int]) -> str:
    s = tuple(int(x) for x in shape)
    if len(s) == 3 and s[0] == 1 and s[2] >= 6 and s[1] <= 2000:
        return "Possible NMS/detection rows [1, N, >=6]"
    if len(s) == 3 and s[0] == 1 and s[1] in (4, 5, 6, 7, 84, 85) and s[2] >= 100:
        return "Possible YOLO raw head (channels-first style [1,C,N])"
    if len(s) == 3 and s[0] == 1 and s[2] in (4, 5, 6, 7, 84, 85) and s[1] >= 100:
        return "Possible YOLO raw head (rows style [1,N,C])"
    if len(s) == 4:
        return "4D feature-map output (could be intermediate or SSD-style heads)"
    if len(s) == 2 and s[1] >= 6:
        return "Possible flat detection output [N, >=6]"
    return "Unknown/non-standard detection output shape"


def compare_scenario_outputs(scenario_outputs: dict[str, list[np.ndarray]]) -> list[str]:
    notes: list[str] = []
    keys = list(scenario_outputs.keys())
    if not keys:
        return notes
    num_out = len(scenario_outputs[keys[0]])
    for out_idx in range(num_out):
        arrs = {k: scenario_outputs[k][out_idx] for k in keys}
        # identical output check
        all_identical = True
        for a_name, b_name in itertools.combinations(keys, 2):
            if not np.array_equal(arrs[a_name], arrs[b_name]):
                all_identical = False
                break
        if all_identical:
            notes.append(
                f"output[{out_idx}]: IDENTICAL for black/white/noise -> suspicious (model may be dead/wrong)."
            )
            continue

        # mean absolute deltas
        deltas = []
        for a_name, b_name in itertools.combinations(keys, 2):
            a = arrs[a_name].astype(np.float64, copy=False).ravel()
            b = arrs[b_name].astype(np.float64, copy=False).ravel()
            mad = float(np.mean(np.abs(a - b))) if a.size else 0.0
            deltas.append((a_name, b_name, mad))

        if all(mad < 1e-6 for _, _, mad in deltas):
            notes.append(
                f"output[{out_idx}]: near-zero inter-scenario deltas -> suspiciously input-insensitive."
            )
        else:
            best = ", ".join([f"{a}/{b} MAD={mad:.6g}" for a, b, mad in deltas])
            notes.append(f"output[{out_idx}]: inter-scenario deltas: {best}")
    return notes


def run() -> int:
    parser = argparse.ArgumentParser(
        prog="tflite_raw_output_probe",
        description="Raw tflite_runtime output probe for EdgeTPU/int8 debugging",
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, type=str)
    parser.add_argument(
        "--use-tpu",
        action="store_true",
        help="Try loading EdgeTPU delegate",
    )
    parser.add_argument(
        "--tpu-lib",
        type=str,
        default=DEFAULT_TPU_LIB,
        help="EdgeTPU delegate library path/name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for random-noise synthetic input",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print stats only (skip full raw tensor array dumps)",
    )
    parser.add_argument(
        "--input-real-space",
        choices=("auto", "normalized", "pixels"),
        default="auto",
        help=(
            "For quantized integer inputs: normalized=[0,1], pixels=[0,255], "
            "auto=run both and compare responsiveness."
        ),
    )
    parser.add_argument(
        "--raw-json-out",
        type=str,
        default="",
        help="Optional path to write tensor stats/metadata JSON.",
    )
    args = parser.parse_args()

    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError as exc:
        raise SystemExit(
            "tflite_runtime is required for this script.\n"
            "Install on target with: pip install tflite-runtime"
        ) from exc

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model file not found: {model_path}")

    delegates = None
    if args.use_tpu:
        print(f"[INFO] Trying EdgeTPU delegate: {args.tpu_lib}")
        try:
            delegates = [load_delegate(args.tpu_lib)]
        except Exception as exc:
            raise SystemExit(f"Failed to load EdgeTPU delegate '{args.tpu_lib}': {exc}")

    print(f"[INFO] Loading model: {model_path}")
    if delegates is not None:
        interpreter = Interpreter(model_path=str(model_path), experimental_delegates=delegates)
    else:
        interpreter = Interpreter(model_path=str(model_path))

    interpreter.allocate_tensors()
    print("[INFO] allocate_tensors() successful.")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print_tensor_detail_block("INPUT TENSORS", input_details)
    print_tensor_detail_block("OUTPUT TENSORS", output_details)

    print("\n=== QUANTIZATION SANITY ===")
    for i, d in enumerate(input_details):
        for line in quant_sanity_for_detail(d, f"input[{i}]"):
            print(line)
    for i, d in enumerate(output_details):
        for line in quant_sanity_for_detail(d, f"output[{i}]"):
            print(line)

    if len(input_details) != 1:
        print(
            f"[WARN] Model has {len(input_details)} input tensors. This probe only drives input[0] directly."
        )
    in0 = input_details[0]
    shape = tuple(int(x) for x in in0["shape"])
    if len(shape) != 4:
        raise SystemExit(
            f"Only 4D input tensors supported by this probe; got input[0].shape={shape}"
        )
    _, h, w, c = shape
    print(f"\n[INFO] Synthetic input shape inferred from model: H={h}, W={w}, C={c}")

    synthetic = make_synthetic_inputs(h=h, w=w, c=c, seed=args.seed)

    def run_for_mode(mode: str) -> tuple[dict[str, list[np.ndarray]], dict[str, list[dict[str, Any]]]]:
        scenario_outputs: dict[str, list[np.ndarray]] = {}
        scenario_stats: dict[str, list[dict[str, Any]]] = {}
        print(f"\n=== INPUT REAL-SPACE MODE: {mode} ===")
        for scenario_name, image_u8 in synthetic.items():
            print(f"\n=== RUNNING SCENARIO: {scenario_name.upper()} ===")
            tensor_in = convert_image_for_input(
                image_u8=image_u8,
                input_detail=in0,
                input_real_space=mode,
            )
            interpreter.set_tensor(in0["index"], tensor_in)
            interpreter.invoke()

            outs: list[np.ndarray] = []
            out_stats: list[dict[str, Any]] = []
            for out_i, out_d in enumerate(output_details):
                arr = interpreter.get_tensor(out_d["index"])
                outs.append(arr.copy())

                stats = tensor_stats(arr)
                out_stats.append(stats)
                print(f"output[{out_i}] shape={arr.shape} dtype={arr.dtype}")
                print(
                    f"  min={stats['min']:.8g} max={stats['max']:.8g} "
                    f"mean={stats['mean']:.8g} std={stats['std']:.8g} "
                    f"nonzero_ratio={stats['nonzero_ratio']:.6f}"
                )
                if not args.summary_only:
                    # Print the full raw array as requested.
                    with np.printoptions(threshold=np.inf, linewidth=160):
                        print(f"  RAW output[{out_i}] array:\n{arr}")

            scenario_outputs[scenario_name] = outs
            scenario_stats[scenario_name] = out_stats
        return scenario_outputs, scenario_stats

    if args.input_real_space == "auto":
        modes = ["normalized", "pixels"]
    else:
        modes = [args.input_real_space]

    mode_outputs: dict[str, dict[str, list[np.ndarray]]] = {}
    mode_stats: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for mode in modes:
        outputs, stats = run_for_mode(mode)
        mode_outputs[mode] = outputs
        mode_stats[mode] = stats

    print("\n=== OUTPUT SHAPE INTERPRETATION ===")
    for out_i, out_d in enumerate(output_details):
        msg = classify_detection_like_shape(out_d["shape"])
        print(f"output[{out_i}] shape={tuple(int(x) for x in out_d['shape'])}: {msg}")

    print("\n=== CROSS-SCENARIO SENSITIVITY CHECK ===")
    mode_scores: dict[str, float] = {}
    for mode in modes:
        print(f"[{mode}]")
        notes = compare_scenario_outputs(mode_outputs[mode])
        for line in notes:
            print(line)
        score = 0.0
        for out_idx in range(len(output_details)):
            pair_mads: list[float] = []
            scenarios = mode_outputs[mode]
            for a_name, b_name in itertools.combinations(scenarios.keys(), 2):
                a = scenarios[a_name][out_idx].astype(np.float64, copy=False).ravel()
                b = scenarios[b_name][out_idx].astype(np.float64, copy=False).ravel()
                mad = float(np.mean(np.abs(a - b))) if a.size else 0.0
                pair_mads.append(mad)
            if pair_mads:
                score += max(pair_mads)
        mode_scores[mode] = score

    if len(mode_scores) > 1:
        ranked = sorted(mode_scores.items(), key=lambda kv: kv[1], reverse=True)
        print("\n=== AUTO MODE RECOMMENDATION ===")
        for mode, score in ranked:
            print(f"{mode}: responsiveness_score={score:.8g}")
        print(f"recommended_mode={ranked[0][0]}")

    print("\n=== QUICK DIAGNOSIS ===")
    flat_notes: list[str] = []
    for out_i in range(len(output_details)):
        all_zero_like = True
        for mode in modes:
            for scenario_name in mode_outputs[mode]:
                s = mode_stats[mode][scenario_name][out_i]
                if abs(s["max"]) > 1e-8 or abs(s["min"]) > 1e-8:
                    all_zero_like = False
                    break
            if not all_zero_like:
                all_zero_like = False
                break
        if all_zero_like:
            flat_notes.append(f"output[{out_i}] is all-zero across all test inputs.")
    if flat_notes:
        for n in flat_notes:
            print(f"[SUSPECT] {n}")
        print(
            "[SUSPECT] This strongly suggests wrong model file, broken graph, or delegate/model incompatibility."
        )
    else:
        print("[INFO] Outputs are not trivially all-zero across all synthetic inputs.")

    if args.raw_json_out:
        payload = {
            "model_path": str(model_path),
            "used_tpu_delegate": bool(args.use_tpu),
            "input_real_space_mode": args.input_real_space,
            "input_details": [{k: _to_python(v) for k, v in d.items()} for d in input_details],
            "output_details": [{k: _to_python(v) for k, v in d.items()} for d in output_details],
            "scenario_stats": mode_stats,
            "mode_scores": mode_scores,
            "shape_interpretation": [
                classify_detection_like_shape(d["shape"]) for d in output_details
            ],
            "cross_scenario_notes": {
                mode: compare_scenario_outputs(mode_outputs[mode]) for mode in modes
            },
        }
        out_path = Path(args.raw_json_out).expanduser().resolve()
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote JSON report: {out_path}")

    print("\n[INFO] Probe complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
