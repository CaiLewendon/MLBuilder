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

## 2026-05-07/08 Addendum: Local Live + Gimbal Simulation Baseline

### Key fixes completed
1. Core parser fix in `MLBuilder/model/tflite/tflitemodel.py`:
   - corrected NMS-path bbox decoding for models outputting raw-head style tensors
   - corrected `cv2.dnn.NMSBoxes` input to `xywh`
2. Label usage standardized in local live test script:
   - defaults to `target_detector_labels.txt`
3. Gimbal simulation script finalized and renamed:
   - `test/tf_live_infrence_gimbal_simulation.py`

### Known-good command to reuse
```bash
venv/bin/python test/tf_live_infrence_gimbal_simulation.py --video 0
```

### Why this command is preferred
- Best observed local performance in current session.
- Center-crop pass active by default (critical to recall).
- Confidence default set to `0.20`.
- Same model family and label path proven during session.

### Next technical milestone
- Move from print-only gimbal command simulation to actual MAVLink transmission and verify real gimbal motion correctness (direction sign, clamp behavior, response smoothness).

## 2026-05-12 Addendum: The "Wrong Model Deployed" Diagnosis

### Executive summary
After weeks of suspected quantization bugs, calibration problems, training-quality issues, and preprocessing regressions, the real cause turned out to be that **the Pi was loading a different int8 EdgeTPU model than the one we believed**. The model in production was never the `project1_prod` April-trained artifact. The code, the wrapper, the quantization math, and even the calibration set were all fine for the artifact actually deployed — but the artifact itself was the wrong one.

### How the diagnosis fell out

1. User reported `<0.005` confidences and "model stopped working" on the Pi.
2. Initial hypothesis (wrong): the Pi `tflitemodel.py` was missing the input-quantization fix.
3. User pasted the Pi's actual `tflitemodel.py` — it already had the proper input quantization and output dequantization, with the strict `[N,6]` branch gate.
4. Added diagnostic prints to wrapper (`[TFLITEMODEL] Loaded from:`, ALLOCATE block) and ran on Pi.
5. Live output `[OUT] shape=(1, 5, 8400) max=1.0073 mean=0.244` and stable 0.13-confidence detections proved the wrapper and dequant were healthy.
6. Hashed the two int8 EdgeTPU artifacts on the Pi:
   - `~/model_full_integer_quant_edgetpu.tflite` → `e4623d5d...`
   - `~/target_detector_int8_edgetpu.tflite`     → `153b25f3...`
7. Cross-referenced against repo artifacts:
   - `e4623d5d...` = `export/model_full_integer_quant_edgetpu.tflite` (March 13 — OLD)
   - `153b25f3...` = four byte-identical copies in repo: `target_detector_int8_edgetpu.tflite` (May 1), `export/project1_prod_full_integer_quant_edgetpu.tflite` (Apr 16), `export/project1_prod_int8_edgetpu_compat.tflite` (Apr 16), `export/pi_rebuild_edgetpu/target_detector_int8_edgetpu.tflite` (May 10)
8. Probed non-edgetpu siblings to characterize each model:
   - OLD: input `(0.01866, -14)`, output `(1,5,8400)` raw-head, `(0.00403, -126)`.
   - NEW: input `(0.00392, -128)`, output `(1,300,6)` postprocessed, `(0.00402, -122)`.
9. Conclusion: production has been running the pre-`project1_prod` March model the entire time. The newer model trained on the 492-image dataset has copies on the Pi but is not the one the script defaults to.

### Critical artifact identification table

| Hash prefix | Build date | Architecture | Trained on |
|---|---|---|---|
| `e4623d5d` | March 13 | raw-head `(1,5,8400)` | pre-project1_prod (legacy) |
| `153b25f3` | April 16 / May 1 | postprocessed `(1,300,6)` w/ NMS | project1_prod, 492-image dataset |

### Why earlier symptoms map to "wrong model"

- "Confidences <0.005 on the Pi" — likely an earlier observation against a different model file at that path, or a sub-threshold view of the OLD model in a different scene.
- "Model stopped working" episodes — file-mtime evidence shows the deployed `~/model_full_integer_quant_edgetpu.tflite` got overwritten at 2026-05-12 00:29:36 (same second as `tflitemodel.py`). The overwrite kept the hash `e4623d5d...` (no actual content change), but the wrapper update at the same second is what made the system behave consistently after.
- "Clutter false positive at 0.13, weak far recall" — exactly the OLD model's behavior on a scene it was never trained for.
- "Close target works at >0.50" — even a poorly-matched model can hit obvious targets.

### Wrapper status (unchanged, verified correct)
- `MLBuilder/model/tflite/tflitemodel.py` reads quant params from the model file at allocate time and applies the right transform on both ends.
- Branch gate cleanly separates raw-head `[5,8400]` and postprocessed `[N,6]` outputs by `6 <= shape[1] <= 16 AND shape[0] > shape[1]`.
- Same wrapper handles both OLD and NEW model files without modification — no code change needed when swapping the deployed model.
- Diagnostic prints retained: `[TFLITEMODEL] Loaded from:` (module load), `[ALLOCATE] ...` (allocate time). Useful for any future "which file is loaded" question.

### Things ruled out as causes (do not chase again)
- `tflitemodel.py` input or output quantization bug — already fixed.
- `(1,5,8400)` vs `[N,6]` branch routing — already correct.
- 128×128 calibration NPY — never used by the deployed builds.
- Video pipeline resolution mismatch — wrapper letterboxes internally.

### Action items resulting from this session
1. Swap deployed Pi model to the `153b25f3` artifact (already on the Pi at `~/target_detector_int8_edgetpu.tflite`).
2. Add model-hash printing at script startup so any future deployment mismatch is visible immediately.
3. Add regression tests for both decode branches.
4. Treat `export/project1_prod_int8_edgetpu_compat.tflite` as the canonical repo source of the production model (user designation).
