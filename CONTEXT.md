# Deployment Context (Deep State Snapshot)

## Operating Topology
- Source camera stream:
  - RTSP `rtsp://10.42.0.1:8554/front_high`
- Pi:
  - consumes RTSP, runs inference with Coral TPU
  - emits processed stream to downstream relay path
- Relay:
  - receives Pi output
  - forwards to laptop
- Laptop:
  - receives relay output via `gst-launch-1.0`

## Runtime Environments
- Repo workspace (laptop dev): `~/Documents/aerospace2025-26/MLBuilder`
- Production runtime (Pi): `~/MLBuilder`
- Production inference script filename: `tf_live_inferenceV2.py` (Pi-local; not tracked in repo history shown here)

## Active Model/Label Artifacts
- Edge TPU model used on Pi:
  - `~/model_full_integer_quant_edgetpu.tflite`
- Labels:
  - `../target_detector_labels.txt` (from script working directory)
- Class label expected:
  - `Target`

## Proven Working Baselines
### Pi inference launch
```bash
python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o
```

### Laptop receive (when relay is up and sending H264 on 5000)
```bash
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

## Critical Model Output Fact
- Pi raw inference output probe:
  - output tensor shape: `(1, 5, 8400)`
  - dequantized output max ~`1.007`
- Interpretation:
  - raw-head detection format, not `[N,6]` postprocessed format.

## Critical Production Code Fix (Must Persist)
- In `TFLiteModel.detect()`:
  - old: `if out.ndim == 2 and out.shape[1] >= 6:`
  - required: `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:`
- Rationale:
  - ensures `(5,8400)` goes to raw-head decode path.

## Current Behavior
- Inference path:
  - functional
  - prints detections when filters are permissive enough
- Quality:
  - can mislabel foreground clutter as target
  - can miss background/far target

## Current Script Feature Set (as iterated in session)
- optional post-filtering:
  - min confidence
  - max area ratio
  - edge-touching suppression
- optional crop passes:
  - center crop pass
  - optional second crop guidance discussed
- optional contrast preprocessing (CLAHE) discussed for low-contrast far target recall

## Known Failure Modes
1. Relay down:
   - no laptop video despite valid inference.
2. Over-filtering:
   - `raw > 0` but `filtered = 0`.
3. Under-filtering:
   - higher false positives in clutter.

## Resume Checklist
1. Verify relay up.
2. Verify Pi TPU allocation lines show active `True`.
3. Run with debug counts:
   - raw candidate count
   - filtered count
4. Tune only one axis at a time:
   - confidence thresholds
   - crop geometry
   - filter thresholds
