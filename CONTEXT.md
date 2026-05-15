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

## 2026-05-13 Session Addendum (FullDataSetProd: 3,487-image retrain + export)

### Headline
New, larger-dataset successor to `project1_prod` is trained and exported through int8 TFLite. Only the EdgeTPU compile step remains before Pi deployment. Pi production currently still runs OLD March model (`e4623d5d`); FullDataSetProd is intended to supersede both that and the project1_prod `153b25f3` artifact.

### Dataset
- Source: `project-1-at-2026-05-13-06-40-8e81e090/` — fresh Label Studio "YOLO with Images" export
- **3,487 images** (7× the prior 492-image `project1_prod` dataset), single class `Target`, 1920×1080
- 858 background-only images (no `Target` instance) — 24.6% explicit negatives
- Original camera/source clip identity is NOT preserved in Label Studio filenames (UUID prefix + frame-number suffix only)

### Scene-aware split (prevents val leakage from near-duplicate video frames)
- Tool: `test/prepare_dataset_split.py` (numpy + PIL only, no extra deps)
- Method: dhash (difference perceptual hash, 64-bit) → union-find by Hamming distance ≤ 5 → largest-first balanced bin packing to hit 80/20
- Result: **2,790 train / 697 val** (exact 80/20), 1,026 clusters, 802 singletons
- One unusually large cluster of **990 near-identical images** (~28% of dataset) was kept whole in train. Likely a near-stationary camera sequence. Implication: val (697 images, ~all other scenes) is a **pessimistic benchmark** — harder than typical production scenes.

### Training (yolo11n base, matches project1_prod hyperparams)
- Command:
  ```bash
  venv/bin/yolo detect train model=yolo11n.pt \
    data=project-1-at-2026-05-13-06-40-8e81e090/data.yaml \
    imgsz=640 epochs=40 batch=20 \
    project=build/out name=FullDataSetProd
  ```
- Resolved hyperparams: AdamW, lr=0.002, momentum=0.9, AMP, default augmentations
- Wall time: ~16 min on RTX 3070 Laptop (~26s/epoch), ~9 GB GPU memory peak
- **Final val: P=0.956, R=0.85, mAP50=0.951, mAP50-95=0.719**

### ⚠️ Gotcha: Ultralytics ignored project= argument
- Ultralytics' `~/.config/Ultralytics/settings.json` has a `runs_dir` that overrode `project=build/out` — output landed in `/home/caile/Documents/MLBuilder/runs/detect/build/out/FullDataSetProd/` (note: NOT in `aerospace2025-26/`).
- Promoted manually: `cp /home/caile/Documents/MLBuilder/runs/detect/build/out/FullDataSetProd/weights/best.pt export/FullDataSetProd.pt`

### Export saga (OOM + dependency fix)
- **First failure**: `AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'` — `onnx_graphsurgeon` 0.5.8 incompatible with installed `onnx` 1.20.1. Fix: `pip install --upgrade onnx_graphsurgeon` → 0.6.1.
- **Second failure (OOM)**: int8 calibration with the full 3,487-image set materializes the entire calibration tensor in memory (~17 GB at imgsz=640×640×3 float32). On a 16 GB laptop with VS Code open, systemd-oomd killed both yolo AND the Chromium process (VS Code crashed at the same time as the export). Two consecutive yolo OOM kills at 13 GB RSS confirmed in journalctl.
- Fix: 500-image random subset (`project-1-at-2026-05-13-06-40-8e81e090/calib_subset_500.txt`, seeded `shuf`), referenced by `data_calib_subset.yaml`. Comfortably above Ultralytics' "300+ images" recommendation, peak ~3 GB calibration RAM. Successful export in ~35 min wall time.

### Final artifacts (in `/home/caile/Documents/aerospace2025-26/MLBuilder/export/`)

| File | Size | sha256 (full) | Input dtype | Output dtype | Output shape |
|---|---|---|---|---|---|
| `FullDataSetProd.pt` | 5.45 MB | `f213fdf4b4a30bf97ebc561a0148638a707b12c1015842053a925e0c3495e971` | (pytorch) | (pytorch) | `(1, 5, 8400)` |
| `FullDataSetProd_last.pt` | 5.45 MB | `a464d96d832743c94842095e42512caf6bbfd688821b93eec9c3e2dfc6714d56` | (pytorch) | (pytorch) | `(1, 5, 8400)` |
| `FullDataSetProd.onnx` | 10.69 MB | `547ec1ce0f1f0a9d90fa8dbda998b69c8a90d31d31bba6dfbd6af525743a5129` | float32 | float32 | `(1, 5, 8400)` |
| `FullDataSetProd_saved_model/FullDataSetProd_float32.tflite` | 10.61 MB | `424173e5a932a818ec4e79da9dd0cdf4f3056eceea8944d1ab0851dc38198a04` | float32 | float32 | `(1, 5, 8400)` |
| `FullDataSetProd_saved_model/FullDataSetProd_float16.tflite` | 5.36 MB | `5e5e795e1e3d1687a230f462342511d111a0007629f3341ea2017e6fb6c72f05` | float32 | float32 | `(1, 5, 8400)` |
| `FullDataSetProd_saved_model/FullDataSetProd_integer_quant.tflite` | 2.94 MB | `e7a68aec7575b30ecda0b1ff942f48db656766165cb46cd3a1d2996a322affcc` | float32 | float32 | `(1, 5, 8400)` |
| `FullDataSetProd_saved_model/FullDataSetProd_int8.tflite` | 2.98 MB | `392783403883bc090237dbf93a0634be075b6bdf3393aa20acd9fb4decb8a281` | float32 | float32 | `(1, 5, 8400)` |
| **`FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant.tflite`** | **2.94 MB** | **`db579abd1359952f48b9cb4e83c3ffaf3836c080e683bb04c25b8fa6b87f1534`** | **int8** | **int8** | **`(1, 5, 8400)`** |

**Canonical EdgeTPU input** = `FullDataSetProd_full_integer_quant.tflite` (the only variant with int8 IO — required for the EdgeTPU compiler).
- Input quantization: scale=`0.00392`, zero=`-128` (≈ standard 1/255 normalization)
- Output quantization: scale=`0.003909`, zero=`-128`

### ⚠️ Critical: output-shape signature changed from prior canonical
- FullDataSetProd has output shape **`(1, 5, 8400)`** — raw head, NOT the postprocessed `(1, 300, 6)` of the canonical `project1_prod` (`153b25f3`).
- This is because `nms=False` was passed to Ultralytics export (matches `test/rebuild_tpu_model_pi.py`'s default).
- **Consequence**: On the Pi, FullDataSetProd will print `[OUT] shape=(1, 5, 8400)` — the same as the OLD broken March model (`e4623d5d`). The visual deployment fingerprint that previously distinguished `(1, 300, 6) = good` from `(1, 5, 8400) = bad` no longer works.
- The wrapper (`MLBuilder/model/tflite/tflitemodel.py`) handles `(1, 5, 8400)` correctly via its raw-head decode branch — no code change needed.
- **Deployment-identity policy going forward**: use `sha256sum` of the deployed model, not output shape. Future production startup print should emit hash.
- Open decision (logged in KANBAN): re-export with `nms=True` to restore `(1, 300, 6)` signature for fingerprinting parity, or keep raw-head and rely on hash.

### What still needs to happen on the laptop (for full handoff to Pi)
1. Install `edgetpu_compiler` (not on laptop yet):
   ```bash
   curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
   sudo apt update && sudo apt install edgetpu-compiler
   ```
2. Compile:
   ```bash
   edgetpu_compiler -s -o export/FullDataSetProd_saved_model/ \
     export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant.tflite
   ```
   Produces `FullDataSetProd_full_integer_quant_edgetpu.tflite` in same dir.
3. Pi deployment (BACK UP old file first):
   ```bash
   ssh pi 'cp ~/target_detector_int8_edgetpu.tflite ~/target_detector_int8_edgetpu.tflite.PROJECT1PROD_153b25f3_BACKUP'
   scp export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_edgetpu.tflite \
       pi@<PI>:~/FullDataSetProd_edgetpu.tflite
   ```
4. Run on Pi (new file path):
   ```bash
   python3 -B tf_live_inferenceV2.py ~/FullDataSetProd_edgetpu.tflite --tpu \
     -l ../target_detector_labels.txt -p -o
   ```
   Expected `[OUT] shape=(1, 5, 8400)` (raw head). Verify hash of deployed file matches `db579abd...` truncated/post-compile, recorded after compile.

### Pipeline learnings (record for future)
- `onnx_graphsurgeon` version must keep up with `onnx` version — bump pre-emptively if `onnx.helper.float32_to_bfloat16` style errors appear
- For Ultralytics int8 calibration on consumer hardware: cap calibration set at ~500 images. More doesn't materially help quant; less than 300 is below Ultralytics' recommendation.
- Ultralytics' `~/.config/Ultralytics/settings.json` overrides `project=` — set it explicitly OR `unset` it before training.

## 2026-05-13 Session Addendum (Manual Gimbal Control + Live Telemetry CLI)

### Headline
`test/manual_gimbal_control.py` is working well. It is the operator/bench tool for driving the gimbal directly and verifying MAVLink link health on the Pi. Same command shape as `tf_live_infrence_gimbal_live.py`; intent is to deploy and run **on the Pi** (not the laptop).

### What it is
- Continuous, single-keypress gimbal teleop (cbreak terminal mode — no Enter required)
- Live, ANSI-refreshing telemetry panel for confirming the autopilot link is alive
- Pure MAVLink — no camera, no inference, no GStreamer; only requires Python 3.9 + `pymavlink`

### Control conventions (identical to `tf_live_infrence_gimbal_live.py`)
- MAVLink message: `MAV_CMD_DO_MOUNT_CONTROL` (cmd 205) via `command_long_send`
- `param1` = pitch deg (+ up, − down), `param3` = yaw deg (+ right, − left)
- `param7` = `MAV_MOUNT_MODE_MAVLINK_TARGETING` (mode 2)
- Limits: yaw ±90°, pitch ±45° (clamped client-side before send)
- **Position-based** absolute angle command — script accumulates a setpoint locally and re-sends the absolute angle each tick

### Keymap (single keypress, OS auto-repeat = continuous motion while held)
- `w` / `s` — pitch up / down by `step`
- `a` / `d` — yaw left / right by `step`
- `c` — center (pitch=0, yaw=0)
- `+` / `-` — increase / decrease step (range 0.5–30°)
- `r` — force resend of current setpoint
- `q` (or Ctrl-C) — quit

### Telemetry surface (drained via `recv_match(blocking=False)`)
Each row shows last-seen value + age, prefixed with `+` (fresh <2s) / `!` (stale) / `X` (never seen):
- `HEARTBEAT` — confirms link, autopilot type, base/custom mode, system_status
- `ATTITUDE` — drone roll/pitch/yaw (deg)
- `SYS_STATUS` — battery V/I/%
- `GLOBAL_POSITION_INT` — lat/lon/alt/relalt/hdg
- `VFR_HUD` — airspeed/groundspeed/alt/climb/throttle/heading
- `GPS_RAW_INT` — fix_type/sats/eph/epv
- `MOUNT_STATUS` — autopilot's reported gimbal pitch/roll/yaw (this is the feedback to verify the command actually moved the gimbal)
- `GIMBAL_DEVICE_ATTITUDE_STATUS` — MAVLink v2 quaternion → euler (alternate feedback path)
- `RC_CHANNELS` — chan1-8 raw + rssi
- `COMMAND_ACK` — confirms our `MAV_CMD_DO_MOUNT_CONTROL` is being accepted
- `STATUSTEXT` — autopilot messages

### Rates
- Telemetry stream request: `MAV_DATA_STREAM_ALL` at 10 Hz on connect
- Gimbal command send: up to 20 Hz on change + 2 Hz background resend (heartbeat) so the autopilot can't time out
- Panel redraw: 10 Hz

### Deployment
This is the kind of file you **do** scp to the Pi (unlike inference scripts that live in the repo on the laptop):
```bash
# from laptop, in repo root
scp test/manual_gimbal_control.py pi@10.42.0.1:~/
```

### Known-good run commands (on Pi)
```bash
python3 ~/manual_gimbal_control.py                                # default tcp:10.42.0.1:5760
python3 ~/manual_gimbal_control.py --mavlink udpin:0.0.0.0:14550  # alt endpoint
python3 ~/manual_gimbal_control.py --no-mavlink                   # offline panel test (no transmit)
```

### CLI flags
- `--mavlink <conn>` — pymavlink connection string (default `tcp:10.42.0.1:5760`)
- `--no-mavlink` — print-only dry-run
- `--heartbeat-timeout` — seconds to wait for first heartbeat (default 15)
- `--stream-rate` — requested telemetry stream rate Hz (default 10)
- `--send-rate` — max gimbal command rate Hz (default 20)
- `--heartbeat-send-rate` — background setpoint-resend Hz (default 2)
- `--redraw-rate` — telemetry panel refresh Hz (default 10)
- `--step` — degrees per keypress (default 2.0)
- `--initial-pitch` / `--initial-yaw` — starting angles

### Gimbal vs drone control conventions (clarification recorded here for future)
- **Gimbal** = absolute position (`MAV_CMD_DO_MOUNT_CONTROL` sends absolute pitch/yaw degrees). What both `tf_live_infrence_gimbal_live.py` and this manual CLI do.
- **Drone** = velocity-based (vector). `tf_live_infrence_drone_simulation.py` computes body-frame `vx` / `vz` / `yaw_rate` from yaw-error, lidar stand-off-error, and altitude-pixel-error. No real MAVLink sends in the sim — it integrates the velocities into an internal lidar/altitude model. When taken live, the natural translation is `SET_POSITION_TARGET_LOCAL_NED` in `MAV_FRAME_BODY_OFFSET_NED` with a type_mask that ignores position+accel and honors `vx/vy/vz` + `yaw_rate`.

### Python compatibility note
File uses `from __future__ import annotations` so `float | None` style PEP-604 union hints evaluate as strings. Confirmed to parse under Python 3.9 grammar (Pi runs 3.9.19).

## 2026-05-14 — Autonomous gimbal automation (production)

### File
`test/tf_live_inferenceV2_gimbal_auto.py` — single production script. User-validated end-to-end (quote: "works amazingly").

### Integration sources (combined, not refactored — each script remains)
- V2 threaded inference (`test/tf_live_inferenceV2.py`)
- Simulation gain law (`test/tf_live_infrence_gimbal_simulation.py`)
- Manual-gimbal MAVLink plumbing (`test/manual_gimbal_control.py`)

### Threading model
| Thread | Role | Trigger | Rate |
|---|---|---|---|
| `capture_thread` | RTSP → `latest_frame[0]`, drop-old | continuous | camera-bound (~30 fps source) |
| `inference_thread` | latest_frame → detections → setpoint via `link.update_setpoint(pitch, yaw)` | `frame_event` + seq-dedup | TPU-bound (~7 fps on Pi) |
| `gimbal_tx_thread` | `MAV_CMD_DO_MOUNT_CONTROL` send | dirty bit OR heartbeat tick | `--send-rate` (20 Hz) when dirty / `--heartbeat-send-rate` (2 Hz) otherwise |
| `gimbal_rx_thread` | drain inbox, log ACK/STATUSTEXT | blocking recv_match | autopilot-driven |
| main | optional video writer | every frame | 10 fps when `--no-output` not set |

### Control law (exact)
| Step | Formula | Notes |
|---|---|---|
| Target center | `cx=(x1+x2)/2`, `cy=(y1+y2)/2` | from max-confidence bbox |
| Normalized error | `err_x=(cx-W/2)/(W/2)`, `err_y=(cy-H/2)/(H/2)` | range `[-1, +1]`, resolution-independent |
| Deadband | `\|err_x\| < 0.08 AND \|err_y\| < 0.08` → NO UPDATE | settles loop when target ~centered |
| Gain (yaw) | `current_yaw += yaw_gain * err_x` | `yaw_gain=12` default; target-right → yaw-clockwise+ |
| Gain (pitch) | `current_pitch -= pitch_gain * err_y` | `pitch_gain=10` default; image +y is down → subtract for gimbal +up |
| Clamp | yaw `[-90, +90]`, pitch `[-45, +45]` | matches manual-gimbal limits |
| MAVLink | `command_long(205, p1=pitch, p3=yaw, p7=2)` | `MAV_MOUNT_MODE_MAVLINK_TARGETING` |

### Sign conventions (printed at startup, for verification)
- image: +x=right, +y=down, center=(W/2, H/2)
- gimbal yaw (param3): +right/clockwise, -left/counter-clockwise
- gimbal pitch (param1): +up, -down
- control: target right → yaw+, target left → yaw-, target down → pitch-, target up → pitch+

### `--center-gimbal` mode
Skips camera/TFLite/threading entirely. Connects MAVLink, sends `(pitch=0, yaw=0)` `--center-duration` × `--center-rate` packets (defaults 2s × 5 Hz = 10 packets), then exits. Honors `--no-mavlink` for dry-run.

### Default args that work
| Arg | Default | Why |
|---|---|---|
| `--deadband` | `0.08` | 8% of half-width = noise floor for jittery detections |
| `--yaw-gain` | `12.0` | heuristic; conservative for stability without FOV calibration |
| `--pitch-gain` | `10.0` | heuristic; slightly lower than yaw because pitch limit is tighter |
| `--send-rate` | `20.0` Hz | fast enough that tracking lag is detection-bound not transmit-bound |
| `--heartbeat-send-rate` | `2.0` Hz | matches `manual_gimbal_control.py`; keeps gimbal driver from timing out |
| `--stream-rate` | `10` Hz | telemetry inbox rate request; matches manual-control |
| `--heartbeat-timeout` | `15.0` s | wait for first FC heartbeat on connect |

### Outstanding (next session)
- **Gain tuning** — defaults are heuristic. FOV-calibrated mapping is `deg_err = pixel_err × HFOV/2`; deadbeat tracking wants `yaw_gain ≈ HFOV/2`, `pitch_gain ≈ VFOV/2`. Without HFOV/VFOV captured, leaving as-is.
- **Gimbal feedback** — currently open-loop on MAVLink ACK only. `MOUNT_STATUS` and `GIMBAL_DEVICE_ATTITUDE_STATUS` are received and logged by `run_rx_loop` but not used for closed-loop verification.
- **PWM fallback** — user asked about `MAV_CMD_DO_SET_SERVO`-based centering, then redirected. Pitch/yaw servo channels and neutral PWM values not yet captured.
- **Earth-frame stabilization** — not compensating airframe roll/pitch from `ATTITUDE`. Body-frame only.

---

## 2026-05-14 Late Session Addendum (Slew-rate-limited control + state-machine firing)

### Gimbal control law — final architecture (replaces earlier "cumulative-P" snapshot above)
The cumulative-P integrator (`current_yaw += yaw_gain * err_x`) was diagnosed as the root cause of overshoot on real hardware. Cumulative additions outran the physical gimbal slew rate; by the time the gimbal had moved a few degrees, the integrator had commanded tens of degrees ahead. The fix is a **two-tier control system** owned by `GimbalLink`:

| Tier | Owned by | Update rate | What it does |
|---|---|---|---|
| **Controller** (`wished_yaw/pitch`) | inference thread | ~5 Hz (per detection) | `wished_yaw = link.get_current() + yaw_gain * err_x` — computed FRESH each frame against the actual transmitted position. NOT integrated. |
| **Transmitter** (`current_yaw/pitch`) | `run_tx_loop` in GimbalLink | 20 Hz | Each tick, step `current_*` toward `wished_*` by at most `max_slew_rate_* / 20` degrees. Hard rate limit. Transmits the (slowly-moving) `current_*` via `MAV_CMD_DO_MOUNT_CONTROL`. |

Because inference reads `current_yaw` (the slow-moving transmitted value) to compute `wished_yaw`, the integrator naturally stops growing as the gimbal physically moves and `err_x` shrinks. Standard proportional-against-actual control with a slew-rate-limited setpoint. No accumulation, no runaway.

**Defaults (CLI args):**
- `--yaw-gain 12.0`, `--pitch-gain 10.0` — same as simulation script (intentional; we want sim-equivalent gain).
- `--max-slew-rate-yaw 2.0`, `--max-slew-rate-pitch 1.5` (deg/sec) — conservative, micro-stepping at 20 Hz: 0.10° / 0.075° per tick. Convergence on a 30° world-angle error takes ~15 s. Raise for faster tracking when gimbal slew capability is known.
- `--deadband 0.08` — unchanged.

**Per-frame log carries both setpoints:**
```
[F0001] target_bbox=((735,330),(1155,780)) center=(945,555) err=(-0.016,+0.028)
  conf=0.918 dir=CENTERED cur=(y+1.13,p+37.71) wished=(y+1.13,p+37.71)
  fire_phase=IDLE CMD_LONG cmd=205 param1=37.71 ...
```
`cur` = where the gimbal physically is (transmitted). `wished` = controller intent. They converge when err→0.

### Startup orientation
- **Default**: script centers the gimbal first by calling `center_gimbal()` (sends `(0,0)` for `center_duration` × `center_rate` packets), then begins tracking from `(0,0)`.
- **`--start-from-current`**: script reads `MOUNT_STATUS` or `GIMBAL_DEVICE_ATTITUDE_STATUS` from the autopilot for up to `--read-current-timeout` seconds via `read_current_gimbal_position()` and uses the reported angles as the initial reference. Gimbal does not move on startup. Aborts cleanly if no message received in time. **Confirmed working** — user's gimbal reported `pitch=+37.71 yaw=+1.13` and tracking continued smoothly from that reference.

### Fire control — final architecture
Phase state machine, **`RELAY_STATUS`-confirmed transitions**. Replaces all earlier attempts (DO_REPEAT_RELAY double-send, DO_SET_RELAY ON+timer+OFF, idle heartbeats, blind keepalive, closed-loop verify).

**Phases:**
- `IDLE`: relay confirmed OFF (or not-yet-fired). Waiting for CENTERED target. 2 Hz `DO_SET_RELAY OFF` keepalive.
- `ARMING`: sent `DO_SET_RELAY ON`. 2 Hz reassertion. Transitions to FIRING only when `RELAY_STATUS` reports relay bit = 1.
- `FIRING`: relay confirmed ON. Timer counts `fire_period` seconds from confirmed-ON instant. Continued 2 Hz `DO_SET_RELAY ON` reassertion in case of mid-burst dropouts.
- `DISARMING`: sent `DO_SET_RELAY OFF`. 2 Hz reassertion. Transitions to COOLDOWN only when `RELAY_STATUS` reports relay bit = 0.
- `COOLDOWN`: relay confirmed OFF. `fire_cooldown` seconds idle gap. 2 Hz `DO_SET_RELAY OFF` reassertion. Returns to IDLE when timer expires, resetting `lock_fired`.

**Commands used:**
- `MAV_CMD_DO_SET_RELAY` (181) for the state changes. Idempotent — sending ON when already ON does nothing; sending OFF when already OFF does nothing.
- `MAV_CMD_SET_MESSAGE_INTERVAL` (511) on connect to subscribe `RELAY_STATUS` (msg 376) at 5 Hz.
- `MAV_CMD_DO_MOUNT_CONTROL` (205) for gimbal — unchanged.
- `MAV_CMD_DO_REPEAT_RELAY` (182) is defined and `send_repeat_relay()` exists but is **not called** by the autonomous fire path anymore (used only in `manual_gimbal_control.py --live-fire` one-shot).

**Fire CLI flags:**
- `--live-fire` — actually send real `DO_SET_RELAY` commands. Without this, runs in BLANK mode (logs phase transitions, no real fire).
- `--no-fire` — disables fire logic entirely (no logs, no commands).
- `--fire-relay 1` — relay instance. Matches QGroundControl "Shoot Gun" action default.
- `--fire-period 5.0` — seconds ON per burst.
- `--fire-cooldown 0.5` — seconds OFF between back-to-back bursts (auto-repeat). Set to 0 for one-shot-per-lock (must un-center to re-fire).

**Phase visible in every per-frame log:** `fire_phase=IDLE|ARMING|FIRING|DISARMING|COOLDOWN`.

**Phase-transition logs (one per transition):**
```
[LIVE ARMING] sent DO_SET_RELAY(1,1); awaiting RELAY_STATUS confirmation that bit[1]=1
[LIVE FIRE ON CONFIRMED] RELAY_STATUS confirmed after 0.34s; holding for 5.00s
[LIVE BURST DONE] held ON for 5.01s; sent DO_SET_RELAY(1,0); awaiting OFF confirmation
[LIVE FIRE OFF CONFIRMED] RELAY_STATUS confirmed after 0.12s; cooldown 0.50s before next burst
[LIVE FIRE READY] cooldown complete; re-armed for next centering event
```

### Manual gimbal control script — `--live-fire` re-purposed as one-shot action
`test/manual_gimbal_control.py --live-fire` now bypasses the interactive panel entirely:
1. Connect MAVLink, wait heartbeat
2. Send ONE `MAV_CMD_DO_REPEAT_RELAY` (matching QGC "Shoot Gun" action exactly: `param1=1 param2=1 param3=2`)
3. Drain inbox up to ~2.5 s looking for `COMMAND_ACK` → print `[ACK] cmd=182 result=N`
4. Close connection, exit

This was the **isolation harness** that proved the relay command works correctly when issued from a single linear code path with no concurrent threads, no panel repaint loop, no keyboard auto-repeat. Confirmed firing on a single CLI invocation. Used as the validation step before trusting the same control flow inside the autonomous script.

Interactive panel (without `--live-fire`): gimbal control + BLANK fire simulation only.

### Confirmed during this session
- Gimbal tracking with slew-rate limit converges smoothly without overshoot at `2.0/1.5 deg/sec` defaults on actual hardware.
- `--start-from-current` correctly reads gimbal orientation from `MOUNT_STATUS` / `GIMBAL_DEVICE_ATTITUDE_STATUS` and starts tracking from that reference.
- `DO_SET_RELAY` ON/OFF pattern fires the gun reliably (relay 1, gun trigger).
- `RELAY_STATUS` is published by the autopilot when subscribed via `SET_MESSAGE_INTERVAL`.
- Full autonomous loop: target detection → centered → ARMING → FIRING (5s) → DISARMING → COOLDOWN → ready for next centering event.
- Exit safety always sends `DO_SET_RELAY OFF` in the `finally` block (cleanup on Ctrl-C, crash, normal exit).

### Next session priorities
1. **Tune `fire_period` and `fire_cooldown`** for the actual mission requirements (currently 5s fire / 0.5s rest).
2. **Tune slew rates** — `2.0/1.5 deg/sec` is conservative. If smoothness is good, raise toward `5-10 deg/sec` for faster target re-acquisition.
3. **Field test on airframe** — confirm tracking behavior in flight (vibration, attitude changes, lighting).
4. **Decide on live-inference output choke** (open from prior session): `--no-output` works for autonomous-only; H.264 encoded relay or two-machine split needed if laptop viewer is required.
5. **Consider FOV calibration** for deadbeat tracking (still open from prior session).
6. **Earth-frame stabilization** (still open).

---

## 2026-05-14 Late Addendum #2 (Fire path refactored: DO_REPEAT_RELAY + COMMAND_ACK-gated)

After completing the DO_SET_RELAY ON+timer+OFF + RELAY_STATUS-gated state machine (the previous addendum) and confirming it worked, we refactored the fire path to use `DO_REPEAT_RELAY` (matching QGC's "Shoot Gun" command shape) with `COMMAND_ACK` as the confirmation source instead of `RELAY_STATUS`. The new path is simpler — autopilot owns the relay-pulse timing, the script just sends one command and tracks elapsed time.

### Why we changed it
- User requested: *"can we switch it to the do repeat relay with the same arming and disarming logic?"*
- First attempt re-used `RELAY_STATUS` as the confirmation gate → **got stuck in ARMING forever**. `DO_REPEAT_RELAY(cycles=1)` toggles ON→OFF→back, so RELAY_STATUS reflects the *commanded final* state (which equals the initial state) and may never show intermediate ON.
- User identified the right next step: *"we should also be able to check the ack command from do repeat relay to verify the states no?"* — yes, COMMAND_ACK fires within ~100ms of the autopilot accepting the command and is the reliable truth source for "did the autopilot get the command?".

### Final architecture (4 phases)

```
IDLE ──[centered]──► ARMING ──[ACK result=0]──► FIRING ──[fire_period sec]──► COOLDOWN ──[2*fire_period + fire_cooldown sec total]──► IDLE
                       │ │
                       │ └─[ACK result≠0]──► IDLE (rejected — log error, no cooldown needed)
                       │
                       └─[1.0s timeout, no ACK]──► FIRING (assume command got through, ACK lost)
```

**Phase details:**

| Phase | Entry action | Exit condition |
|---|---|---|
| IDLE | None | Target enters CENTERED → send ONE `DO_REPEAT_RELAY(relay=1, cycles=1, period=2*fire_period)`, record `fire_send_time = now`, → ARMING |
| ARMING | (none) | (1) `COMMAND_ACK cmd=182 result=0` after `fire_send_time` → FIRING. (2) `COMMAND_ACK cmd=182 result≠0` after `fire_send_time` → IDLE (rejected). (3) `now - fire_send_time >= 1.0s` without matching ACK → FIRING (timeout-assumed) |
| FIRING | (none) | `now - fire_send_time >= fire_period` → COOLDOWN |
| COOLDOWN | (none) | `now - fire_send_time >= 2*fire_period + fire_cooldown` → IDLE |

**MAVLink traffic per fire cycle:**
- Outbound: 1× `DO_REPEAT_RELAY` (one packet). That's it.
- Inbound: 1× `COMMAND_ACK` cmd=182. Plus the always-on inbound `RELAY_STATUS` at 5 Hz (autopilot publishes; we just listen for log visibility).
- **No keepalive. No reassertion. No periodic OFF commands.** Minimum possible command rate.

### GimbalLink additions for this iteration

```python
class GimbalLink:
    # ... existing ...
    self._last_repeat_relay_ack_result = None  # int, e.g. 0=ACCEPTED, 2=DENIED, 4=FAILED, 5=UNSUPPORTED
    self._last_repeat_relay_ack_time = None    # time.monotonic() when ACK was captured

    def get_last_repeat_relay_ack(self) -> tuple:
        """Returns (result, monotonic_timestamp_or_None)."""
        with self._lock:
            return self._last_repeat_relay_ack_result, self._last_repeat_relay_ack_time
```

`run_rx_loop` now captures the ACK when handling `COMMAND_ACK` for `cmd=182`:
```python
if cmd == MAV_CMD_DO_REPEAT_RELAY:
    with self._lock:
        self._last_repeat_relay_ack_result = int(msg.result)
        self._last_repeat_relay_ack_time = time.monotonic()
```

### What was REMOVED from the previous iteration
- `ARMING` and `DISARMING` phases gated on `RELAY_STATUS` (replaced by single ACK-gated ARMING; DISARMING no longer needed since autopilot auto-OFFs after the cycle's ON half).
- 2 Hz `DO_SET_RELAY` keepalive in every phase (replaced by zero keepalive — autopilot owns timing).
- `fire_state["last_send"]`, `fire_state["lock_fired"]` (no longer needed with the simpler timer model).
- `fire_state["phase_start_time"]` (replaced by `fire_send_time` which serves all phase timing).

### What was KEPT from the previous iteration
- `RELAY_STATUS` subscription at 5 Hz via `MAV_CMD_SET_MESSAGE_INTERVAL` (511) on connect. Still useful for visibility.
- `--live-fire`, `--no-fire`, `--fire-relay`, `--fire-period`, `--fire-cooldown` CLI flags (semantics preserved).
- BLANK mode auto-confirmation (now 0.1 s simulated ACK).
- Exit safety `DO_SET_RELAY OFF` in `finally` block (interrupts in-progress cycle on Ctrl-C / crash).
- `run_rx_loop` printing `[ACK-RELAY] cmd=182 result=N (0=ACCEPTED, ...)` lines.

### CLI flag semantics

- `--fire-period 5.0` (default) — user-facing "ON time" in seconds. Script sends `param3 = 2 * fire_period = 10.0` so ArduPilot's cycle-half is 5s.
- `--fire-cooldown 0.5` (default) — ADDITIONAL idle after the autopilot's full cycle (2*fire_period) completes, before next fire is allowed. So total time between fires = `2 * fire_period + fire_cooldown = 10.5s` with defaults.
- `--fire-relay 1` (default) — matches QGC "Shoot Gun" action's `param1=1`.
- `--live-fire` — actually send `DO_REPEAT_RELAY`. Without it, BLANK mode logs phase transitions only.
- `--no-fire` — disables fire logic entirely.

### Per-frame log signature

Successful cycle:
```
[F0001] ... dir=CENTERED ... fire_phase=IDLE ...
[LIVE ARMING] sent DO_REPEAT_RELAY(1,cycles=1,period=10.00s); awaiting COMMAND_ACK for cmd=182
[F0002] ... fire_phase=ARMING ...
[ACK-RELAY] cmd=182 result=0 (0=ACCEPTED, 4=FAILED, 2=DENIED, 5=UNSUPPORTED)
[LIVE ARMED] COMMAND_ACK result=0 (ACCEPTED) after 0.18s; autopilot will pulse relay ON for ~5.00s
[F0003] ... FIRING_LIVE_t=0.21s ...
[F0050] ... FIRING_LIVE_t=4.95s ...
[LIVE BURST DONE] 5.00s ON elapsed; RELAY[1]=OFF; awaiting cycle completion + 0.50s cooldown
[F0051] ... fire_phase=COOLDOWN ...
[LIVE FIRE READY] cycle (10.00s) + cooldown (0.50s) complete; RELAY[1]=?; re-armed
```

Rejected ACK:
```
[LIVE ARMING] sent DO_REPEAT_RELAY(1,cycles=1,period=10.00s); awaiting COMMAND_ACK for cmd=182
[ACK-RELAY] cmd=182 result=4 ...
[LIVE ARMING REJECTED] COMMAND_ACK result=4 (2=DENIED, 4=FAILED, 5=UNSUPPORTED) after 0.18s; NOT firing; back to IDLE
```

Timeout (no ACK):
```
[LIVE ARMING] sent DO_REPEAT_RELAY(1,cycles=1,period=10.00s); awaiting COMMAND_ACK for cmd=182
[LIVE ARMED WARN] no COMMAND_ACK in 1.00s; assuming command got through (ACK may have been lost). If no fire occurred, autopilot may have rejected silently.
```

### Status: NEEDS PI RE-VALIDATION

The previous iteration (DO_SET_RELAY + RELAY_STATUS-gated) was user-validated working. This refactor to DO_REPEAT_RELAY + COMMAND_ACK-gated has been compile-tested only. **The next Pi run should confirm it works end-to-end before this is called "final".**

Quickest validation: pass `--live-fire` and watch for the `[ACK-RELAY] cmd=182 result=0` line. If you see it, the cycle is running and the rest of the state machine is timer-driven so will progress.

### Next session priorities (updated)
1. **Pi re-validation of the ACK-gated fire path** — confirm `[ACK-RELAY] cmd=182 result=0` arrives and the cycle behaves as expected. If it doesn't, the DO_SET_RELAY version is in git history (one commit back).
2. Tune `fire_period` / `fire_cooldown` for mission.
3. Tune slew rates.
4. Field test on airframe.
5-7. (other open items from previous addendum unchanged).
