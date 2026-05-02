# Deployment Context (Current Ground Truth)

## Environment
- Laptop dev repo: `~/Documents/aerospace2025-26/MLBuilder`
- Production runtime: Raspberry Pi + Coral Edge TPU
- Inference script in production: `tf_live_inferenceV2.py` (on Pi)

## Model + Labels
- Edge TPU deployment model:
  - `~/model_full_integer_quant_edgetpu.tflite` (on Pi)
- Labels:
  - `../target_detector_labels.txt` from script directory

## Known Good Run Command (Pi)
```bash
python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o -c 0.01
```

## Confirmed Allocation Output
- TPU active: `True`
- Input dtype: `int8`
- Input shape: `[1, 640, 640, 3]`

## Critical Decode Fact
- Raw output format from deployed model:
  - `out shape: (5, 8400)` (raw head)
- It is **not** `[N,6]` postprocessed output for this deployed artifact.

## Critical Code Fix (Production)
- In `TFLiteModel.detect()`, changed `[N,6]` branch gate from:
  - `if out.ndim == 2 and out.shape[1] >= 6:`
- to:
  - `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:`

This ensures `(5,8400)` takes raw-head/NMS path instead of incorrect postprocessed path.

## Symptoms Before Fix
- Inference loop advanced (`infer_frames=...`) but `dets=0` forever.

## Symptoms After Fix
- Non-zero detections reported continuously in production logs.

## Relay/Viewing Context
- One “no display” episode was due to relay not running, not model inference failure.
- Keep inference logs and relay pipeline debugging separate during triage.
