# Kanban

## Done
- Established Pi + Coral inference bring-up end-to-end.
- Confirmed correct model artifact for TPU deployment.
- Diagnosed and fixed raw-head decode routing bug (`(5,8400)` vs `[N,6]`).
- Verified relay outage as independent failure source from inference.
- Added first-pass runtime quality mitigations:
  - geometric post-filtering
  - optional crop-pass strategy
- Verified detections can be produced continuously after decode fix.
- Fixed TFLite bbox conversion/NMS bug in core parser:
  - normalized `xywh` handling corrected
  - `cv2.dnn.NMSBoxes` input corrected to `xywh` format
- Confirmed local live detection baseline is stable with center crop pass.
- Added and validated gimbal simulation script:
  - `test/tf_live_infrence_gimbal_simulation.py`
  - print-only `MAV_CMD_DO_MOUNT_CONTROL` command output
  - axis conventions displayed in console
- Set labels default to `target_detector_labels.txt` in live test path.
- Added and iterated drone positioning simulation script:
  - `test/tf_live_infrence_drone_simulation.py`
  - yaw-only center alignment (horizontal axis)
  - forward/back stand-off control with inner-band trim toward exact setpoint
  - post-approach altitude placement phase (target to bottom-quarter frame goal)
  - final active hold with disturbance rejection (`yaw + vx + vz`)
  - clarified on-screen vectors (single yaw vector + single altitude-to-goal vector)
  - confidence gating and control telemetry refinements

## In Progress
- **CRITICAL: Deploy correct int8 model to Pi production**
  - Pi has been running OLD March 13 model (`e4623d5d...`), NOT the project1_prod April model (`153b25f3...`).
  - The correct model already exists on the Pi at `~/target_detector_int8_edgetpu.tflite` and `~/project1_prod_int8_edgetpu_compat.tflite` — production just points at the wrong file.
  - Two-line swap on Pi to fix (see CONTEXT.md "Production deployment fix").
  - Verification: `[OUT] shape` should change from `(1, 5, 8400)` → `(1, 300, 6)`.
- Drone test integration planning:
  - map simulation control outputs (`yaw_rate`, `vx`, `vz`) to real MAVLink commands
  - validate sign conventions and gain scaling on airframe/simulator
  - preserve current simulation overlays/logs as debug parity harness

## Done (2026-05-12 Session)
- Identified that two distinct int8 EdgeTPU models exist:
  - `e4623d5d...` (OLD, March 13, raw-head `(1,5,8400)`, pre-project1_prod training).
  - `153b25f3...` (NEW, April project1_prod, postprocessed `(1,300,6)`, 492-image dataset).
- Cross-referenced Pi-deployed hashes against repo artifacts — confirmed Pi was running OLD model.
- Confirmed `MLBuilder/model/tflite/tflitemodel.py` correctly handles both architectures (no code change needed; wrapper reads quant params from model file).
- Added permanent diagnostic prints to wrapper (`[TFLITEMODEL] Loaded from`, `[ALLOCATE] ...`).
- Verified output quant of OLD model on Pi (live run): max ≈ 1.007, mean ≈ 0.244, confidences in expected range — model works as well as it can on its trained data, just trained on wrong data.
- Identified canonical "best" copy of the new model per user: `export/project1_prod_int8_edgetpu_compat.tflite`.
- Built calibration helper artifacts on laptop:
  - `project-1-at-2026-04-12-21-16-9fb8c3ae/calib_all.txt` (492 train+val image paths)
  - `project-1-at-2026-04-12-21-16-9fb8c3ae/data_calib.yaml` (val key points at calib_all.txt)

## Next (High Priority)
1. **Swap model on Pi + verify** — see CONTEXT.md.
2. After swap, capture metrics from a real target session:
   - close-target confidence
   - far-target confidence vs clutter background
   - whether bbox is sticky (clutter lock) or tracks a moving target
3. Run drone-motion simulation soak test with recorded video and live camera sources.
4. Implement optional real-command path behind a flag for:
   - yaw control
   - forward/back distance control
   - altitude hold correction
5. Validate end-to-end sign conventions and units against simulator/airframe.
6. Keep center-crop pass enabled by default for this positioning path.
7. Save one locked command profile for field testing.

## Next (Medium Priority)
1. Add temporal confirmation gate (2-of-3 frame persistence) as optional argument.
2. Add CLI flags for all filter thresholds and crop centers/ratios.
3. Add a "debug overlay mode" to color boxes by source pass (full/crop1/crop2).

## Backlog
1. Retraining path for durable accuracy improvement (only AFTER confirming new-model behavior on Pi):
   - more far-distance positives
   - hard-negative clutter examples (specifically the static clutter that the OLD model locked onto)
2. Rebuild int8 with 492-image calibration (currently 99 — only val split):
   - artifacts ready: `project-1-at-2026-04-12-21-16-9fb8c3ae/calib_all.txt` and `data_calib.yaml`
   - blocked by: `edgetpu_compiler` not installed on laptop; install via Coral apt repo (see CONTEXT.md / install runbook)
   - run: `venv/bin/python test/rebuild_tpu_model_pi.py --data project-1-at-2026-04-12-21-16-9fb8c3ae/data_calib.yaml`
3. Add regression tests for decoder branch selection in `TFLiteModel.detect()` covering both `(1,5,8400)` and `(1,300,6)` outputs.
4. Add deployment health checks:
   - model-hash validation at script startup (warn if hash doesn't match expected production model)
   - relay alive check
   - receiver caps consistency check
5. Add a model-identity print at startup (`sha256` of loaded model) so a wrong-deployment incident is immediately visible.

## Blockers / Risks
- Single-class model in cluttered scene has intrinsic ambiguity at long distance.
- Over-filtering currently causes all detections to disappear in some configs.
- Lighting and perspective variation likely exceed model robustness without retraining.
- Real autopilot response/gain tuning may differ from simulation assumptions.
- **Two int8 model artifacts share a similar-looking name (`*_full_integer_quant_edgetpu.tflite`) but are completely different models** — high deployment-confusion risk. Mitigation: prefer the explicit name `target_detector_int8_edgetpu.tflite` / `project1_prod_int8_edgetpu_compat.tflite` going forward.

## Known-Good Commands (Current Session)
1. Live detection baseline (laptop):
   - `venv/bin/python test/tf_live_infrence.py export/project1_prod_saved_model/project1_prod_float16.tflite --video 0 --center-crop-pass --center-crop-ratio 0.5 -c 0.20 --log-detections`
2. Best gimbal simulation baseline (laptop):
   - `venv/bin/python test/tf_live_infrence_gimbal_simulation.py --video 0`
3. Drone positioning simulation baseline (laptop):
   - `venv/bin/python test/tf_live_infrence_drone_simulation.py --video 0`
4. Drone simulation (stronger visible auto-correction tuning, laptop):
   - `venv/bin/python test/tf_live_infrence_drone_simulation.py --video 0 --sim-lidar-start-cm 400 --sim-lidar-approach-factor 3.0 --min-distance-correct-vx 0.12 --deadband 0.12`
5. Pi production with CORRECT model (post-swap):
   - `python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o`
6. Pi probe (TFLite metadata inspection):
   - file at `/tmp/probe.py` on Pi; uses `tflite_runtime` with EdgeTPU delegate.
