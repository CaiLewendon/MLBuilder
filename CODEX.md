# CODEX Session Log: Edge TPU + Relay + Runtime Tuning

Date: 2026-05-02
Project: `MLBuilder`
Primary runtime: Raspberry Pi + Coral Edge TPU
Viewer path: Relay -> laptop GStreamer receiver

## Executive Summary
- Core TPU inference path is operational.
- Relay/viewer path is operational when relay service is up.
- A critical decode bug was fixed in production (`(5,8400)` raw-head routing).
- Current blocker is quality, not bring-up:
  - false positives on foreground objects
  - weak recall on far/background target instances
  - aggressive filtering can suppress all detections if tuned too hard

## Chronological Milestones
1. Located live scripts and model artifacts.
2. Verified dataset split and labels:
   - total 492 images
   - train 393, val 99
3. Identified deployment models:
   - CPU-side test model: `model_saved_model/model_int8.tflite`
   - Edge TPU model: `export/model_full_integer_quant_edgetpu.tflite`
4. Confirmed `_edgetpu.tflite` cannot run on non-TPU interpreter (expected `edgetpu-custom-op` failure on CPU).
5. Confirmed Pi allocation path:
   - TPU active `True`
   - input dtype `int8`
   - input shape `[1,640,640,3]`
6. Isolated decode bug:
   - probe showed output shape `(1,5,8400)` -> raw head format
   - legacy `[N,6]` condition was too broad and consumed raw-head output
7. Applied minimal production fix:
   - changed `[N,6]` branch gate to require `N >> cols` shape
8. Re-ran on Pi:
   - detections resumed (`detections=1..3` observed)
9. Relay/video mismatch incident:
   - temporary "no video" was relay-down state, not inference failure
10. Added runtime enhancements in script iterations:
   - post-filtering (min conf, area ratio, edge-touch suppression)
   - optional center-crop pass
   - optional second crop and CLAHE suggestions
11. Current quality issue:
   - model can still mislabel near foreground object while missing far background target

## Critical Production Fix (Already Known Good)
- In production `MLBuilder/model/tflite/tflitemodel.py` detect logic:

From:
- `if out.ndim == 2 and out.shape[1] >= 6:`

To:
- `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:`

Why:
- prevents `(5,8400)` raw head tensors from being treated as `[N,6]` postprocessed tensors.

## Verified Good Command Baselines
- Pi inference command baseline:
  - `python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o`
- Laptop receiver baseline (when relay outputs H264 on 5000):
  - `gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false`

## Current Observations (Most Recent)
- With decoding fix, detections appear reliably in logs.
- With stricter filtering/cropping, false positives can reduce, but recall can drop to zero.
- Scene has strong clutter + lighting gradients; small far target is near detection limit.

## Tactical Runtime Levers (No Retraining)
- Lower model-stage threshold (`--confidence`) to keep candidate boxes.
- Use moderate post-filter threshold (`--min-conf`) instead of hard cut.
- Multi-crop ROI strategy (center + upper-middle crop) improves far-target recall.
- CLAHE can help contrast but may also amplify noise; must be tuned empirically.

## Known Pitfalls
- Over-filtering can create "no detections" despite working model.
- Aspect ratio / geometry filters can remove true positives if target perspective varies.
- Relay diagnostics can mask inference diagnostics; always confirm both independently.

## Hard Separation of Concerns
- Inference correctness:
  - confirmed by Pi log detections and raw output probes.
- Stream transport correctness:
  - confirmed separately by relay health + matching receiver caps/codec.

## Remaining Work
- Stabilize false-positive/recall tradeoff without retraining.
- Produce one locked "operational profile" (args set) for reliable demo behavior.
- Add a deterministic debug mode that prints:
  - raw candidate count
  - filtered count
  - crop-pass contribution count
