"""Convert FullDataSetProd saved_model to int8 TFLite using TF 2.15
(emits CONV_2D op <= v5, compatible with edgetpu_compiler 16.0)."""
import os, sys, numpy as np
import tensorflow as tf
print(f"TF: {tf.__version__}", flush=True)

SAVED_MODEL = sys.argv[1] if len(sys.argv) > 1 else 'export/FullDataSetProd_saved_model'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_tf215.tflite'
CALIB_NPY = sys.argv[3] if len(sys.argv) > 3 else 'build/calib_500x3x640x640_float32.npy'   # (N, 3, 640, 640) NCHW

# Load calibration data and convert to NHWC (saved_model expects NHWC)
calib = np.load(CALIB_NPY, mmap_mode='r')          # (N, 3, 640, 640)
print(f"calib shape: {calib.shape} dtype: {calib.dtype} min={float(calib[0].min()):.3f} max={float(calib[0].max()):.3f}", flush=True)
N = calib.shape[0]

def representative_dataset():
    for i in range(N):
        x = calib[i]                                # (3, 640, 640)
        x = np.transpose(x, (1, 2, 0))              # (640, 640, 3) NHWC
        x = x[np.newaxis, ...].astype(np.float32)   # (1, 640, 640, 3)
        yield [x]
        if (i + 1) % 100 == 0:
            print(f"  calibrated {i+1}/{N}", flush=True)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("converting...", flush=True)
buf = converter.convert()
with open(OUT, 'wb') as f:
    f.write(buf)
print(f"wrote {OUT} ({len(buf)} bytes)", flush=True)

# inspect output
intp = tf.lite.Interpreter(model_path=OUT)
intp.allocate_tensors()
for det in intp.get_input_details():
    print(f"INPUT  name={det['name']} shape={det['shape']} dtype={det['dtype']} quant={det['quantization']}")
for det in intp.get_output_details():
    print(f"OUTPUT name={det['name']} shape={det['shape']} dtype={det['dtype']} quant={det['quantization']}")
