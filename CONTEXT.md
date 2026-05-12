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

## 2026-05-07/08 Local Session Addendum (Important)
- Primary local goal completed:
  - stable live detection + gimbal-command simulation baseline in repo test scripts.
- Root cause fixed for "detections exist but bbox looks wrong/off-screen":
  - core parser bug in `MLBuilder/model/tflite/tflitemodel.py` NMS path.
  - fixes applied:
    - normalized-vs-pixel `xywh` handling corrected before remap
    - `cv2.dnn.NMSBoxes` input corrected to `xywh` format (was previously passed as `xyxy`)
- Result:
  - valid bbox coordinates are now produced in the live overlay path.

### Current best-performing local command (save this)
```bash
venv/bin/python test/tf_live_infrence_gimbal_simulation.py --video 0
```

### Why this command is the preferred baseline
- Uses default model:
  - `export/project1_prod_saved_model/project1_prod_float16.tflite`
- Uses center-crop pass by default (important for recall).
- Uses tuned detection confidence default (`0.20`).
- Uses label file by default:
  - `target_detector_labels.txt`
- Prints readable gimbal `COMMAND_LONG` intent without requiring active MAVLink link.

### Companion live-debug command (non-gimbal)
```bash
venv/bin/python test/tf_live_infrence.py export/project1_prod_saved_model/project1_prod_float16.tflite --video 0 --center-crop-pass --center-crop-ratio 0.5 -c 0.20 --log-detections
```

### Next resumption target
- Transition from print-only gimbal simulation to real MAVLink transmission during drone/sim testing, while preserving the same detection defaults and center-crop behavior.

## 2026-05-10/11 Local Session Addendum (Drone Positioning Simulation)
- New simulation script added:
  - `test/tf_live_infrence_drone_simulation.py`
- Detection stack intentionally kept aligned with current baseline:
  - default model: `export/project1_prod_saved_model/project1_prod_float16.tflite`
  - center-crop pass enabled by default
  - default labels path: `target_detector_labels.txt`
  - highest-confidence target selection
- Control strategy implemented (simulation-only, print intent only):
  1. Yaw-to-center alignment (horizontal-only)
  2. Forward/back distance control to stand-off setpoint
  3. Post-approach altitude phase to place target at bottom-quarter frame goal
  4. Final active hold (yaw + distance + altitude disturbance rejection)
- Distance behavior:
  - outer tolerance window preserved (`target ± tolerance`, default `225 ± 15 cm`)
  - inner center-band trim added so controller still seeks exact setpoint while inside tolerance
  - minimum correction velocity added for both forward and backward correction visibility
- Altitude behavior:
  - goal line and vector to altitude target are rendered in-frame
  - altitude state transitions: `ALTITUDE_ADJUST` -> `FINAL_HOLD`
  - active hold includes vertical drift rejection simulation
- Overlay/log clarifications added:
  - explicit `yaw_to_center`
  - horizontal yaw movement vector only (`yaw_vec_px`)
  - single altitude vector to desired altitude point (`alt_to_goal_vec_px`) + altitude goal line
  - commands shown as `cmd[yaw_rate,vx,vz]`
  - disturbance telemetry shown for `vx` and `vz`

### Current best local command for continuation
```bash
venv/bin/python test/tf_live_infrence_drone_simulation.py --video 0
```

### Useful tuning command (stronger visible auto-correction)
```bash
venv/bin/python test/tf_live_infrence_drone_simulation.py --video 0 --sim-lidar-start-cm 400 --sim-lidar-approach-factor 3.0 --min-distance-correct-vx 0.12 --deadband 0.12
```

### Next resumption target
- Map this simulation state machine and control gains to real MAVLink position/yaw commands behind a guarded flag while preserving current visual/telemetry debug surfaces.

## 2026-05-12 Session Addendum (Critical: Wrong Model Was Deployed)

### TL;DR of the day
The Pi production was running the WRONG int8 TPU model for weeks. Every "model quality" symptom — clutter false positives, weak far-target recall, <0.005 confidences on some attempts, "model stopped working" episodes — was rooted in a deployed-model mismatch, not in code, not in quantization, not in calibration math.

### The two int8 EdgeTPU models — definitive identification

**OLD model (`e4623d5d13c1f316c82b0f5a50d4079d9cfd7939cbfd197c11667a97068e1003`)**
- Origin: March 13 training run, pre-`project1_prod` dataset.
- Architecture: raw-head output `(1, 5, 8400)` — needs runtime NMS in wrapper.
- Quantization: input `scale=0.01866, zero=-14`; output `scale=0.00403, zero=-126`.
- Laptop path: `export/model_full_integer_quant_edgetpu.tflite`.
- Pi path: `~/model_full_integer_quant_edgetpu.tflite` (this is what production script defaults to).
- Last modified on Pi: 2026-05-12 00:29:36 (overwritten by an scp/rsync but same content as repo).
- Training data: NOT the 492-image `project-1-at-2026-04-12-21-16-9fb8c3ae` dataset. Pre-relabel era.

**NEW model (`153b25f30817c02075f2d8d06b8b5ea83cb0313dd3b705bd57038e6758fa39a2`) — THE GOOD ONE**
- Origin: April `project1_prod` training run, exported Apr 16 + May 1.
- Architecture: postprocessed output `(1, 300, 6)` — NMS baked into model.
- Quantization: input `scale=0.00392, zero=-128` (≈ 1/255 standard); output `scale=0.00402, zero=-122`.
- Calibration: 99 images (val split, per `build_project1_prod_int8.log`); below Ultralytics' >300 recommended threshold but on real project data.
- Identical copies (byte-for-byte) at:
  - `target_detector_int8_edgetpu.tflite` (repo root, May 1)
  - `export/project1_prod_full_integer_quant_edgetpu.tflite` (Apr 16)
  - `export/project1_prod_int8_edgetpu_compat.tflite` (Apr 16) ← **user-designated canonical best**
  - `export/pi_rebuild_edgetpu/target_detector_int8_edgetpu.tflite` (May 10)
- Pi paths where this model already exists: `~/target_detector_int8_edgetpu.tflite`, `~/project1_prod_int8_edgetpu_compat.tflite` (both hash 153b25...).
- Training data: 492-image `project-1-at-2026-04-12-21-16-9fb8c3ae` dataset.

### Production deployment fix (highest priority)
On the Pi, the script defaults to `~/model_full_integer_quant_edgetpu.tflite` (the OLD March model). Two equivalent fixes:

Option A — leave file alone, change command:
```bash
python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu \
  -l ../target_detector_labels.txt -p -o
```

Option B — swap the file (preserves command):
```bash
# On the Pi
cp ~/model_full_integer_quant_edgetpu.tflite ~/model_full_integer_quant_edgetpu.tflite.OLD_march13_e4623d
cp ~/target_detector_int8_edgetpu.tflite     ~/model_full_integer_quant_edgetpu.tflite
```

Verification: a successful swap shows `[OUT] shape=(1, 300, 6)` (not `(1, 5, 8400)`) on startup. That's the visual signature of the postprocessed-output architecture.

### Wrapper status (`MLBuilder/model/tflite/tflitemodel.py`) — VERIFIED CORRECT
- Reads input/output quantization params from the model file (not hard-coded).
- Input preprocessing: normalize pixel `/255`, then `q = round(real/scale + zero)`, clip to dtype, cast. Correct for either model.
- Output dequantization: `(raw - zero) * scale` when output is int8/uint8 and scale > 0. Correct for either model.
- Branch gate distinguishes raw-head vs postprocessed: `6 <= shape[1] <= 16 AND shape[0] > shape[1]` → postprocessed `[N,6]` path; else raw-head `[5,8400]` path.
- Same wrapper file handles both models without modification. This is verified by the 2026-05-12 Pi live run showing healthy outputs from the old model (max ≈ 1.007, mean ≈ 0.244, confidences in 0.13–0.50 range).
- Diagnostic prints added today: `[TFLITEMODEL] Loaded from: ...` at module load; `[ALLOCATE] ...` at allocate time. These are observation-only and should remain.

### Critical observations from the live Pi run (with OLD model loaded)
- Close target: confidence > 0.50 (model works on obvious targets).
- Persistent clutter detection: confidence ~0.13, bbox glued to near-identical coordinates frame after frame — the OLD model locked onto a static scene feature it weakly classified as `Target`.
- Far target: weak recall, often filtered out below threshold.
- This profile matches the project's long-standing "false positive on clutter + weak far-target recall" documented in earlier CODEX entries — because production was the SAME old model the whole time.

### What changed at 2026-05-12 00:29:36 (same timestamp on Pi)
Both `~/model_full_integer_quant_edgetpu.tflite` and `~/MLBuilder/MLBuilder/model/tflite/tflitemodel.py` were modified at the exact same second on the Pi — almost certainly an `scp`/`rsync` deploy from laptop. The wrapper update is what unblocked correct quant handling; the model file overwrite was a no-op in terms of content (same hash as `export/model_full_integer_quant_edgetpu.tflite` from March 13).

### What was NEVER the cause (rule out for future debugging)
- Input/output quantization in `tflitemodel.py` — handled correctly.
- The `(5,8400)` decode branch gate — handled correctly.
- The 128×128 calibration NPY at repo root — never used by the deployed int8 builds. Build log shows real `data.yaml`-based calibration.
- Resolution mismatch in the video pipeline — irrelevant; wrapper letterboxes 1080p → 640×640 internally.
- EdgeTPU compiler op fallback — not investigated but the OLD model dequantizes cleanly to [0,1.017], so head ops are healthy on TPU.

### Known-good Pi production command (after swap to new model)
```bash
python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu \
  -l ../target_detector_labels.txt -p -o
```
Expected `[OUT] shape` line: `(1, 300, 6)`. Expected confidence range on real target: closer to laptop float16 (~0.4–0.8 on close, 0.2+ on far).

### Next resumption target
1. Swap model on Pi (Option A or B above) and validate `[OUT] shape=(1, 300, 6)` and improved real-target confidence.
2. If still weak on far targets after swap: capture hard-negative footage of the false-positive clutter scene and add to training set. Retrain `project1_prod.pt` and re-export.
3. The 492-image calibration rebuild (`test/rebuild_tpu_model_pi.py --data project-1-at-2026-04-12-21-16-9fb8c3ae/data_calib.yaml`) is now lower priority — only after we know the new model's real-world performance. Requires `edgetpu_compiler` on the laptop (not installed yet).
