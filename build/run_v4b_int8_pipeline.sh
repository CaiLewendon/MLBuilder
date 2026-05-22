#!/usr/bin/env bash
# V4b int8 pipeline — mirrors run_v4_int8_pipeline.sh but for V4b weights.
# Reuses V4's depheavy calib (same dataset → same clusters → same calib).
set -euo pipefail
cd /home/caile/Documents/aerospace2025-26/MLBuilder

DS=project-1-at-2026-05-22-04-22-5cace4ec
SM=export/FullDataSetProdV4b_saved_model
RUNS_BEST=/home/caile/Documents/MLBuilder/runs/detect/build/out/FullDataSetProdV4b/weights/best.pt

echo "==> [0/7] promote best.pt -> export/FullDataSetProdV4b.pt"
[ -f "$RUNS_BEST" ] || { echo "MISSING: $RUNS_BEST"; exit 2; }
cp "$RUNS_BEST" export/FullDataSetProdV4b.pt
sha256sum export/FullDataSetProdV4b.pt | tee build/v4b_pt_sha256.txt

echo "==> [1/7] yolo export int8 (~32 min)"
venv/bin/yolo export model=export/FullDataSetProdV4b.pt format=tflite int8=True \
  data=$DS/data_calib.yaml imgsz=640 2>&1 | tee build/v4b_int8_export.log

echo "==> [2/7] reuse V4 calib NPY (per-cluster) — same dataset/clusters"
[ -f build/calib_500x3x640x640_float32_v4.npy ] || { echo "MISSING per-cluster calib NPY"; exit 2; }
[ -f build/calib_500x3x640x640_float32_v4_depheavy.npy ] || { echo "MISSING depheavy calib NPY"; exit 2; }

echo "==> [3/7] TF 2.15 int8 convert (per-cluster) ~5 min"
venv-tf215/bin/python build/convert_int8_tf215.py \
  $SM \
  $SM/FullDataSetProdV4b_full_integer_quant_tf215.tflite \
  build/calib_500x3x640x640_float32_v4.npy 2>&1 | tee build/v4b_tf215_convert.log

echo "==> [4/7] TF 2.15 int8 convert (depheavy) ~5 min"
venv-tf215/bin/python build/convert_int8_tf215.py \
  $SM \
  $SM/FullDataSetProdV4b_full_integer_quant_tf215_depheavy.tflite \
  build/calib_500x3x640x640_float32_v4_depheavy.npy 2>&1 | tee build/v4b_tf215_convert_depheavy.log

echo "==> [5a/7] surgery + downgrade (per-cluster)"
venv/bin/python build/surgery_grouped_to_depthwise.py \
  $SM/FullDataSetProdV4b_full_integer_quant_tf215.tflite \
  $SM/FullDataSetProdV4b_full_integer_quant_dwfix.tflite 2>&1 | tee build/v4b_surgery.log
venv/bin/python build/downgrade_conv2d_version.py \
  $SM/FullDataSetProdV4b_full_integer_quant_dwfix.tflite \
  $SM/FullDataSetProdV4b_full_integer_quant_dwfix_v3.tflite 2>&1 | tee -a build/v4b_surgery.log

echo "==> [5b/7] surgery + downgrade (depheavy)"
venv/bin/python build/surgery_grouped_to_depthwise.py \
  $SM/FullDataSetProdV4b_full_integer_quant_tf215_depheavy.tflite \
  $SM/FullDataSetProdV4b_full_integer_quant_dwfix_depheavy.tflite 2>&1 | tee build/v4b_surgery_depheavy.log
venv/bin/python build/downgrade_conv2d_version.py \
  $SM/FullDataSetProdV4b_full_integer_quant_dwfix_depheavy.tflite \
  $SM/FullDataSetProdV4b_full_integer_quant_dwfix_depheavy_v3.tflite 2>&1 | tee -a build/v4b_surgery_depheavy.log

echo "==> [6/7] smoke test final tflite IO"
venv-tf215/bin/python -c "
import tensorflow as tf
for tag, path in [('percluster', '$SM/FullDataSetProdV4b_full_integer_quant_dwfix_v3.tflite'),
                  ('depheavy',   '$SM/FullDataSetProdV4b_full_integer_quant_dwfix_depheavy_v3.tflite')]:
    i = tf.lite.Interpreter(model_path=path); i.allocate_tensors()
    inp = i.get_input_details()[0]; out = i.get_output_details()[0]
    print(f'{tag}: in={inp[\"shape\"].tolist()} {inp[\"dtype\"].__name__} q={inp[\"quantization\"]} | out={out[\"shape\"].tolist()} {out[\"dtype\"].__name__} q={out[\"quantization\"]}')
"

echo "==> [7/7] edgetpu compile, both variants"
mkdir -p export
( cd export && edgetpu_compiler -s ../$SM/FullDataSetProdV4b_full_integer_quant_dwfix_v3.tflite ) 2>&1 | tee build/v4b_edgetpu_compile.log
( cd export && edgetpu_compiler -s ../$SM/FullDataSetProdV4b_full_integer_quant_dwfix_depheavy_v3.tflite ) 2>&1 | tee build/v4b_edgetpu_compile_depheavy.log

mv -f export/FullDataSetProdV4b_full_integer_quant_dwfix_v3_edgetpu.tflite \
      export/FullDataSetProdV4b_edgetpu.tflite
mv -f export/FullDataSetProdV4b_full_integer_quant_dwfix_depheavy_v3_edgetpu.tflite \
      export/FullDataSetProdV4b_depheavy_edgetpu.tflite

echo "==> [DONE] artifacts:"
sha256sum export/FullDataSetProdV4b_edgetpu.tflite export/FullDataSetProdV4b_depheavy_edgetpu.tflite
ls -lh export/FullDataSetProdV4b_edgetpu.tflite export/FullDataSetProdV4b_depheavy_edgetpu.tflite
