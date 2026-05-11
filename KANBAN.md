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
- Drone test integration planning:
  - map simulation control outputs (`yaw_rate`, `vx`, `vz`) to real MAVLink commands
  - validate sign conventions and gain scaling on airframe/simulator
  - preserve current simulation overlays/logs as debug parity harness

## Next (High Priority)
1. Run drone-motion simulation soak test with recorded video and live camera sources.
2. Implement optional real-command path behind a flag for:
   - yaw control
   - forward/back distance control
   - altitude hold correction
3. Validate end-to-end sign conventions and units against simulator/airframe.
4. Keep center-crop pass enabled by default for this positioning path.
5. Save one locked command profile for field testing.

## Next (Medium Priority)
1. Add temporal confirmation gate (2-of-3 frame persistence) as optional argument.
2. Add CLI flags for all filter thresholds and crop centers/ratios.
3. Add a "debug overlay mode" to color boxes by source pass (full/crop1/crop2).

## Backlog
1. Retraining path for durable accuracy improvement:
   - more far-distance positives
   - hard-negative clutter examples
2. Add regression tests for decoder branch selection in `TFLiteModel.detect()`.
3. Add deployment health checks:
   - relay alive check
   - receiver caps consistency check

## Blockers / Risks
- Single-class model in cluttered scene has intrinsic ambiguity at long distance.
- Over-filtering currently causes all detections to disappear in some configs.
- Lighting and perspective variation likely exceed model robustness without retraining.
- Real autopilot response/gain tuning may differ from simulation assumptions.

## Known-Good Commands (Current Session)
1. Live detection baseline:
   - `venv/bin/python test/tf_live_infrence.py export/project1_prod_saved_model/project1_prod_float16.tflite --video 0 --center-crop-pass --center-crop-ratio 0.5 -c 0.20 --log-detections`
2. Best gimbal simulation baseline:
   - `venv/bin/python test/tf_live_infrence_gimbal_simulation.py --video 0`
3. Drone positioning simulation baseline:
   - `venv/bin/python test/tf_live_infrence_drone_simulation.py --video 0`
4. Drone simulation (stronger visible auto-correction tuning):
   - `venv/bin/python test/tf_live_infrence_drone_simulation.py --video 0 --sim-lidar-start-cm 400 --sim-lidar-approach-factor 3.0 --min-distance-correct-vx 0.12 --deadband 0.12`
