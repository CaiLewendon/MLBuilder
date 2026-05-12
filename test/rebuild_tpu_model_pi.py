#!/usr/bin/env python3
"""
Rebuild EdgeTPU TFLite model from the project1_prod weights and dataset.

Default flow:
1) Export INT8 TFLite using the project dataset YAML (calibration on ~500 images).
2) Compile that INT8 TFLite with edgetpu_compiler.
3) Copy final artifact to a stable filename for easy SCP.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "export" / "project1_prod.pt"
DEFAULT_DATA = ROOT / "project-1-at-2026-04-12-21-16-9fb8c3ae" / "data.yaml"
DEFAULT_OUTDIR = ROOT / "export" / "pi_rebuild_edgetpu"
DEFAULT_OUTPUT_NAME = "target_detector_int8_edgetpu.tflite"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_dataset_images(data_yaml: Path) -> tuple[int, int]:
    fields: dict[str, str] = {}
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k in ("path", "train", "val"):
            fields[k] = v

    missing = [k for k in ("path", "train", "val") if k not in fields]
    if missing:
        raise RuntimeError(f"data.yaml missing required keys: {missing}")

    base = Path(fields["path"])
    train_list = base / fields["train"]
    val_list = base / fields["val"]
    train_count = sum(1 for _ in train_list.open("r", encoding="utf-8") if _.strip())
    val_count = sum(1 for _ in val_list.open("r", encoding="utf-8") if _.strip())
    return train_count, val_count


def newest_int8_tflite(export_dir: Path) -> Path:
    candidates = []
    for p in export_dir.rglob("*.tflite"):
        name = p.name.lower()
        if "int8" in name or "integer_quant" in name or "full_integer_quant" in name:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No INT8 TFLite found under: {export_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rebuild_tpu_model_pi",
        description="Rebuild EdgeTPU TFLite model from project1_prod + ~500-image dataset",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--final-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python3",
        help="Python binary used to invoke Ultralytics export (default: python3)",
    )
    args = parser.parse_args()

    weights = args.weights.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not weights.is_file():
        print(f"[ERROR] Weights not found: {weights}")
        return 2
    if not data_yaml.is_file():
        print(f"[ERROR] Dataset YAML not found: {data_yaml}")
        return 2
    if shutil.which("edgetpu_compiler") is None:
        print("[ERROR] edgetpu_compiler not found in PATH.")
        print("Install first, e.g. on Debian/RPi:")
        print("  sudo apt-get install edgetpu-compiler")
        return 2

    try:
        train_count, val_count = count_dataset_images(data_yaml)
    except Exception as exc:
        print(f"[ERROR] Failed reading dataset list from {data_yaml}: {exc}")
        return 2

    print("[INFO] Rebuild configuration")
    print(f"[INFO] weights: {weights}")
    print(f"[INFO] data:    {data_yaml}")
    print(f"[INFO] imgsz:   {args.imgsz}")
    print(f"[INFO] outdir:  {outdir}")
    print(f"[INFO] dataset images: train={train_count}, val={val_count}, total={train_count + val_count}")

    # Export INT8 TFLite with dataset-based calibration.
    export_cmd = [
        args.python_bin,
        "-m",
        "ultralytics",
        "export",
        f"model={weights}",
        "format=tflite",
        "int8=True",
        f"imgsz={args.imgsz}",
        f"data={data_yaml}",
        "nms=False",
        f"project={outdir}",
        "name=int8_export",
    ]
    run_cmd(export_cmd, cwd=ROOT)

    int8_tflite = newest_int8_tflite(outdir)
    print(f"[INFO] Selected INT8 model: {int8_tflite}")

    # Compile for EdgeTPU.
    compile_dir = outdir / "compiled"
    compile_dir.mkdir(parents=True, exist_ok=True)
    compile_cmd = [
        "edgetpu_compiler",
        "-s",
        "-o",
        str(compile_dir),
        str(int8_tflite),
    ]
    run_cmd(compile_cmd, cwd=ROOT)

    compiled = list(compile_dir.glob("*_edgetpu.tflite"))
    if not compiled:
        print(f"[ERROR] No *_edgetpu.tflite produced in: {compile_dir}")
        return 3
    compiled_path = max(compiled, key=lambda p: p.stat().st_mtime)

    final_path = outdir / args.final_name
    shutil.copy2(compiled_path, final_path)

    print("[SUCCESS] EdgeTPU model rebuilt.")
    print(f"[INFO] compiled: {compiled_path}")
    print(f"[INFO] final:    {final_path}")
    print(f"[INFO] size:     {final_path.stat().st_size} bytes")
    print(f"[INFO] sha256:   {sha256(final_path)}")
    print("")
    print("[SCP EXAMPLE]")
    print(f"scp pi@<PI_IP>:{final_path} ./")
    return 0


if __name__ == "__main__":
    sys.exit(main())
