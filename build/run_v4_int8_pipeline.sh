#!/usr/bin/env bash
# Run the full V4 int8 pipeline after training completes.
# Assumes export/FullDataSetProdV4.pt is already in place.
set -euo pipefail
cd /home/caile/Documents/aerospace2025-26/MLBuilder

DS=project-1-at-2026-05-22-04-22-5cace4ec
SM=export/FullDataSetProdV4_saved_model
RUNS_BEST=/home/caile/Documents/MLBuilder/runs/detect/build/out/FullDataSetProdV4/weights/best.pt

echo "==> [0/7] promote best.pt -> export/FullDataSetProdV4.pt"
[ -f "$RUNS_BEST" ] || { echo "MISSING: $RUNS_BEST"; exit 2; }
cp "$RUNS_BEST" export/FullDataSetProdV4.pt
sha256sum export/FullDataSetProdV4.pt | tee build/v4_pt_sha256.txt

echo "==> [1/7] yolo export int8 (~32 min)"
venv/bin/yolo export model=export/FullDataSetProdV4.pt format=tflite int8=True \
  data=$DS/data_calib.yaml imgsz=640 2>&1 | tee build/v4_int8_export.log

echo "==> [2/7] build calib NPY (per-cluster)"
venv-tf215/bin/python build/build_calib_npy.py \
  $DS/calib_all.txt \
  build/calib_500x3x640x640_float32_v4.npy 2>&1 | tee build/v4_calib_npy.log

echo "==> [3/7] build calib NPY (depheavy)"
venv-tf215/bin/python build/build_calib_npy.py \
  $DS/calib_deployment_heavy_500.txt \
  build/calib_500x3x640x640_float32_v4_depheavy.npy 2>&1 | tee build/v4_calib_npy_depheavy.log

echo "==> [4/7] TF 2.15 int8 convert (per-cluster) ~5 min"
venv-tf215/bin/python build/convert_int8_tf215.py \
  $SM \
  $SM/FullDataSetProdV4_full_integer_quant_tf215.tflite \
  build/calib_500x3x640x640_float32_v4.npy 2>&1 | tee build/v4_tf215_convert.log

echo "==> [5/7] TF 2.15 int8 convert (depheavy) ~5 min"
venv-tf215/bin/python build/convert_int8_tf215.py \
  $SM \
  $SM/FullDataSetProdV4_full_integer_quant_tf215_depheavy.tflite \
  build/calib_500x3x640x640_float32_v4_depheavy.npy 2>&1 | tee build/v4_tf215_convert_depheavy.log

echo "==> [6a/7] surgery + downgrade (per-cluster)"
venv/bin/python build/surgery_grouped_to_depthwise.py \
  $SM/FullDataSetProdV4_full_integer_quant_tf215.tflite \
  $SM/FullDataSetProdV4_full_integer_quant_dwfix.tflite 2>&1 | tee build/v4_surgery.log
venv/bin/python build/downgrade_conv2d_version.py \
  $SM/FullDataSetProdV4_full_integer_quant_dwfix.tflite \
  $SM/FullDataSetProdV4_full_integer_quant_dwfix_v3.tflite 2>&1 | tee -a build/v4_surgery.log

echo "==> [6b/7] surgery + downgrade (depheavy)"
venv/bin/python build/surgery_grouped_to_depthwise.py \
  $SM/FullDataSetProdV4_full_integer_quant_tf215_depheavy.tflite \
  $SM/FullDataSetProdV4_full_integer_quant_dwfix_depheavy.tflite 2>&1 | tee build/v4_surgery_depheavy.log
venv/bin/python build/downgrade_conv2d_version.py \
  $SM/FullDataSetProdV4_full_integer_quant_dwfix_depheavy.tflite \
  $SM/FullDataSetProdV4_full_integer_quant_dwfix_depheavy_v3.tflite 2>&1 | tee -a build/v4_surgery_depheavy.log

echo "==> [7/7] smoke test final tflite IO"
venv-tf215/bin/python -c "
import tensorflow as tf
for tag, path in [('percluster', '$SM/FullDataSetProdV4_full_integer_quant_dwfix_v3.tflite'),
                  ('depheavy',   '$SM/FullDataSetProdV4_full_integer_quant_dwfix_depheavy_v3.tflite')]:
    i = tf.lite.Interpreter(model_path=path); i.allocate_tensors()
    inp = i.get_input_details()[0]; out = i.get_output_details()[0]
    print(f'{tag}: in={inp[\"shape\"].tolist()} {inp[\"dtype\"].__name__} q={inp[\"quantization\"]} | out={out[\"shape\"].tolist()} {out[\"dtype\"].__name__} q={out[\"quantization\"]}')
"

echo "==> [edgetpu compile, both variants]"
mkdir -p export
( cd export && edgetpu_compiler -s ../$SM/FullDataSetProdV4_full_integer_quant_dwfix_v3.tflite ) 2>&1 | tee build/v4_edgetpu_compile.log
( cd export && edgetpu_compiler -s ../$SM/FullDataSetProdV4_full_integer_quant_dwfix_depheavy_v3.tflite ) 2>&1 | tee build/v4_edgetpu_compile_depheavy.log

# Rename to canonical names
mv -f export/FullDataSetProdV4_full_integer_quant_dwfix_v3_edgetpu.tflite \
      export/FullDataSetProdV4_edgetpu.tflite
mv -f export/FullDataSetProdV4_full_integer_quant_dwfix_depheavy_v3_edgetpu.tflite \
      export/FullDataSetProdV4_depheavy_edgetpu.tflite

echo "==> [DONE] artifacts:"
sha256sum export/FullDataSetProdV4_edgetpu.tflite export/FullDataSetProdV4_depheavy_edgetpu.tflite
ls -lh export/FullDataSetProdV4_edgetpu.tflite export/FullDataSetProdV4_depheavy_edgetpu.tflite
