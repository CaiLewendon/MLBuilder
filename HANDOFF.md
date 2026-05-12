# Session Handoff Runbook (Detailed)

Date: 2026-05-02
Owner context: live Pi + Coral deployment with relay to laptop viewer.

## 1) What Is Working Right Now
- TPU inference allocation on Pi is working:
  - `[ALLOCATE] TPU active: True`
  - input shape `[1,640,640,3]`
- Decode-path fix is validated:
  - model output `(1,5,8400)` now routes to raw-head parser.
- Relay/viewer is known to work when relay is up and output caps match receiver.

## 2) What Is Not Stable Yet
- Quality tuning:
  - foreground clutter may be mislabeled as `Target`.
  - far/background true target can be missed.
- Some script configurations over-filter and return no detections.

## 3) Critical Invariant (Do Not Regress)
In production `TFLiteModel.detect()`:
- Keep `[N,6]` branch gate strict:
  - `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:`

If this reverts, Pi may return zero detections again.

## 4) Commands You Need

### Pi inference baseline
```bash
python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o
```

### Laptop receiver baseline (relay must be up)
```bash
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

## 5) Fast Triage Decision Tree

### Case A: No video on laptop
1. Confirm relay process is running.
2. Confirm relay output codec/port still match receiver command.
3. Confirm Pi is actually writing stream to relay input.

### Case B: Video appears, no boxes
1. Check Pi logs for inference heartbeat (`infer_frames=...`).
2. If no heartbeat: inference loop stalled.
3. If heartbeat present:
   - print `raw_count` and `filtered_count`.
   - if `raw_count > 0` and `filtered_count == 0`: filters too strict.
   - if `raw_count == 0`: detection stage or crop strategy too strict.

### Case C: Boxes appear but wrong object selected
1. Lower reliance on single-frame confidence.
2. Add/enable temporal confirmation.
3. Tune geometry/aspect/area filters gradually.
4. Bias crop strategy toward expected target region.

## 6) Recommended Next Experiment Plan
Run 5-minute fixed-scene tests and log counts:
1. Permissive baseline:
   - low model confidence threshold
   - minimal post-filter
2. Add filter thresholds one at a time:
   - min confidence
   - max area ratio
   - edge-touch suppression
3. Add crop pass:
   - center crop
4. Add upper crop:
   - target likely appears in upper middle background
5. Compare:
   - true target hit rate
   - false positive rate
   - time-to-first-detection

## 7) Suggested Debug Metrics to Print Each N Frames
- `raw_count`
- `filtered_count`
- `crop1_count`
- `crop2_count`
- optional median confidence of kept detections

These make tuning objective instead of visual-only.

## 8) File/Artifact Notes
- Current repo has documentation updates:
  - `CODEX.md`
  - `CONTEXT.md`
  - `KANBAN.md`
  - `HANDOFF.md`
- Production script filename in use:
  - `tf_live_inferenceV2.py` on Pi
- Repo script may differ:
  - `test/tf_live_infrence.py` exists for local experimentation.

## 9) Practical Stop Conditions
- Acceptable interim operating point:
  - consistent detection of true target in background in target scenario
  - false positives below operator tolerance
  - no stream interruptions due to relay mismatch

If unreachable with runtime tuning only, move to retraining with more far-target positives and clutter negatives.

---

## 10) 2026-05-07/08 Session Closeout (Local Live + Gimbal Sim)

### What was fixed in this session
1. Fixed core bbox parsing in `MLBuilder/model/tflite/tflitemodel.py`:
   - corrected normalized/pixel `xywh` interpretation in NMS path
   - corrected OpenCV NMS input format to `xywh`
2. Updated live test script for visibility and repeatability:
   - labels loaded from `target_detector_labels.txt`
   - stable overlay/log behavior retained
3. Added/renamed gimbal simulation script:
   - `test/tf_live_infrence_gimbal_simulation.py`
   - print-only MAVLink-style gimbal commands (`MAV_CMD_DO_MOUNT_CONTROL`)
   - axis conventions printed in console

### Best command right now (keep as baseline)
```bash
venv/bin/python test/tf_live_infrence_gimbal_simulation.py --video 0
```

### Why this is the best current baseline
- Uses center-crop pass by default.
- Uses tuned confidence default of `0.20`.
- Uses latest working float16 default model path in-script.
- Uses label file by default.
- Produces stable detections and readable gimbal command outputs without requiring a MAVLink endpoint.

### Next session first task
- Implement and validate real MAVLink transmission for gimbal control on drone/sim while preserving the current detection defaults (especially center-crop pass).

---

## 11) 2026-05-12 Session Closeout (Wrong Model Deployed — Root Cause Found)

### What was actually broken (and fixed)
The Pi production was running the OLD March 13 int8 model (`e4623d5d...`) instead of the April `project1_prod`-trained model (`153b25f3...`). The newer model exists on the Pi already — `~/target_detector_int8_edgetpu.tflite` — but the production script's default model path points at the older artifact. Code, wrapper, quantization math, and calibration set were all fine for the artifact actually loaded — but the artifact itself was the wrong one.

### What is verified correct (do not modify reflexively)
- `MLBuilder/model/tflite/tflitemodel.py`: handles both `(1,5,8400)` raw-head and `(1,300,6)` postprocessed outputs. Reads quant params from the model file dynamically.
- The branch gate invariant: `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:` — still in place, still correct.
- Diagnostic prints in `allocate()`: keep `[TFLITEMODEL] Loaded from:` and `[ALLOCATE] ...` to make future deployments self-identifying.

### Canonical model artifact (per user designation, 2026-05-12)
**`export/project1_prod_int8_edgetpu_compat.tflite`** (hash `153b25f30817c02075f2d8d06b8b5ea83cb0313dd3b705bd57038e6758fa39a2`) is the canonical production model. Byte-identical to three other files in the repo:
- `target_detector_int8_edgetpu.tflite`
- `export/project1_prod_full_integer_quant_edgetpu.tflite`
- `export/pi_rebuild_edgetpu/target_detector_int8_edgetpu.tflite`

This is the model trained on the 492-image `project-1-at-2026-04-12-21-16-9fb8c3ae` dataset. NMS is baked in (output is `(1, 300, 6)`).

### Production command (after model swap on Pi)
```bash
python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu \
  -l ../target_detector_labels.txt -p -o
```

If swap was via file copy (Option B in CONTEXT.md), the original baseline command still works:
```bash
python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu \
  -l ../target_detector_labels.txt -p -o
```

### Visual signature of "right model loaded"
- `[OUT] shape=(1, 300, 6)` — this is the NEW project1_prod model
- `[OUT] shape=(1, 5, 8400)` — this is the OLD March model (do NOT ship)

### Decision tree if confidence is still weak after swap
1. Verify shape line is `(1, 300, 6)`. If not, the swap didn't take.
2. If far targets still weak: it's a training-data problem (small/far target underrepresented in 492-image set). Capture more far-distance positives and hard-negative clutter from the deployment scene. Retrain `project1_prod.pt`, re-export.
3. If false positives on clutter persist: add clutter scenes (target-absent) to training. Same retrain path.
4. Optional intermediate: rebuild int8 with 492-image calibration instead of 99 (val-only). Artifacts ready at `project-1-at-2026-04-12-21-16-9fb8c3ae/data_calib.yaml`; requires `edgetpu_compiler` on laptop.

### Lessons captured for future debugging
- Always hash the deployed model before assuming code or quantization is the cause. The `sha256sum` step took 5 seconds and would have saved this entire investigation.
- Two different models with similar names (`*_full_integer_quant_edgetpu.tflite`) caused the confusion. Prefer the explicit name `target_detector_int8_edgetpu.tflite` going forward.
- Diagnostic prints in the wrapper (model path, quant params) are cheap and high-leverage; keep them.
