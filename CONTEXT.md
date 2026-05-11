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
