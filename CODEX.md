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

## 2026-05-13 Addendum: FullDataSetProd (3,487-image retrain + export)

### Executive summary
Trained a new, larger-dataset successor to `project1_prod` named `FullDataSetProd` from a 3,487-image Label Studio export (7× prior). Final val mAP50=0.951, mAP50-95=0.719. All int8 + float TFLite variants exported. Only the EdgeTPU compile step remains before deployment to the Pi.

### Chronological session events
1. New Label Studio export landed at `project-1-at-2026-05-13-06-40-8e81e090/` (zip 372 MB, 3,487 images, single class `Target`).
2. Verified export format matches prior `project-1-at-2026-04-12-21-16-9fb8c3ae/` (YOLO with Images). Missing `data.yaml` and `train.txt`/`val.txt` — Label Studio doesn't ship those.
3. Filename inspection: each image has unique UUID prefix; no clip-identifying suffix. Random split would risk near-duplicate video-frame leakage.
4. Wrote `test/prepare_dataset_split.py` — dhash (numpy + PIL only) → union-find clustering → balanced bin packing. Initial run at Hamming threshold 8 over-chained clusters (greedy fill assigned 38% to val). Tightened to threshold 5 + largest-first balanced assignment → exact 80/20 split, 1,026 clusters, largest 990-image cluster kept whole in train.
5. Trained `FullDataSetProd` matching project1_prod hyperparams (yolo11n, imgsz=640, batch=20, epochs=40, AdamW lr=0.002, AMP).
6. Training wall-time ~16 min. Per-epoch mAP50 climbed from 0.63 (epoch 1) to plateau around 0.91–0.93 by epoch 13. Final reported val: P=0.956, R=0.85, mAP50=0.951, mAP50-95=0.719.
7. **Surprise**: Ultralytics ignored `project=build/out` argument. Output landed at `/home/caile/Documents/MLBuilder/runs/detect/build/out/FullDataSetProd/` (different dir entirely). Cause: `~/.config/Ultralytics/settings.json` `runs_dir` setting overrode CLI arg. Worked around by promoting `best.pt` to `export/FullDataSetProd.pt`.
8. **Failure #1: onnx_graphsurgeon import-time crash**. Symptom: int8 export silently exited after labels.cache scan with no error in the log. Root cause via interactive `import onnx2tf`: `AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'`. `onnx_graphsurgeon` 0.5.8 used a function removed in newer `onnx` versions. Fix: `pip install --upgrade onnx_graphsurgeon` → 0.6.1.
9. **Failure #2: OOM**. After fix #1, int8 export crashed again. journalctl showed `Out of memory: Killed process yolo total-vm:33GB, anon-rss:13GB` — twice. Same OOM event killed Chromium (VS Code), which the user observed as "VS Code crashed too." Root cause: Ultralytics materializes the entire int8 calibration set in memory as float32 numpy tensors — 3,487 × 640 × 640 × 3 × 4 bytes ≈ 17 GB on a 16 GB system.
10. Fix: 500-image deterministic random subset of `calib_all.txt` (`shuf --random-source=<(yes 0) -n 500`) → `calib_subset_500.txt` → `data_calib_subset.yaml`. Above Ultralytics' "300+ images" recommendation, ~3 GB calibration RAM peak. Export completed in ~35 min wall time.
11. All TFLite variants produced in `export/FullDataSetProd_saved_model/`. Full hash + IO dtype table in CONTEXT 2026-05-13 addendum.
12. **Critical finding**: Output shape is `(1, 5, 8400)` (raw head) for all FullDataSetProd variants, NOT `(1, 300, 6)` like the prior canonical `project1_prod`. Cause: `nms=False` Ultralytics flag (matches `test/rebuild_tpu_model_pi.py`'s default). The wrapper handles raw-head correctly via its existing branch gate, but the visual deployment fingerprint `(1, 300, 6) = good vs (1, 5, 8400) = bad` no longer holds. Identity must be verified by sha256 hash going forward.

### Lessons captured
- **Always pin `onnx_graphsurgeon` version to track `onnx`** — they release versions in lockstep; mismatch = silent onnx2tf import failure that surfaces as "export hangs after labels.cache scan."
- **Cap int8 calibration set at ~500 images** on consumer 16 GB hardware. Above this, Ultralytics' in-memory tensor allocation triggers OOM. Below 300, you fall under Ultralytics' "recommend more" warning. 500 = sweet spot.
- **Ultralytics' `~/.config/Ultralytics/settings.json` is sticky** — it overrides `project=` CLI args. Either edit it once or always check `find / -name "best.pt"` after a training run to locate output.
- **OOM kills are correlated, not isolated** — a runaway calibration job will take down whatever other RAM-hungry process (browser, IDE) is co-resident in the cgroup. Close those before launching int8 calibration.
- **`nms=False` on export → raw-head output `(1, N_pred, N_anchors)`**; `nms=True` → postprocessed `(1, 300, 6)`. Choose deliberately based on deployment fingerprinting + wrapper expectation.

## 2026-05-14: Autonomous gimbal automation script (USER-VALIDATED, WORKING)

### Executive summary
Built `test/tf_live_inferenceV2_gimbal_auto.py` — a single production script combining all three previously separate concerns: V2 threaded inference, simulation gimbal control law, and manual-gimbal MAVLink plumbing. End-to-end run reported as working ("works amazingly"). Outstanding work is purely tuning (gains + mapping refinement), not architecture.

### What it integrates (per user request: "uses the processing of V2, the gimbal logic of simulation, and the connection of the manual gimbal for the flight computer")
| Source script | What was lifted |
|---|---|
| `test/tf_live_inferenceV2.py` | `pipeline3`/`pipeline4` GStreamer strings, FPS=10/W=1920/H=1080 constants, capture+inference threads with seq-dedup, `--no-output` flag path, `filter_detections`, dual `crop_at` passes, CLAHE preprocessing, `frame_event` + `frame_seq` shared state pattern |
| `test/tf_live_infrence_gimbal_simulation.py` | `clamp`, `direction_label`, deadband math, cumulative-P gain law (`current_yaw += yaw_gain * err_x`, `current_pitch -= pitch_gain * err_y`), gimbal limits `±90° yaw / ±45° pitch`, axis-convention preamble print, log line format `[F######] target_bbox=... CMD_LONG cmd=205 paramN=...` |
| `test/manual_gimbal_control.py` | `pymavlink` connect with `wait_heartbeat(timeout)`, `request_streams(MAV_DATA_STREAM_ALL @ stream_rate)`, separate transmit thread pattern (dirty bit + send-rate + heartbeat-resend-rate), `send_mount_control` exact wire format, `COMMAND_ACK`/`STATUSTEXT` inbox-drain loop |

### Threading architecture (the part the user said works amazingly)
Four daemon threads talking through three locks:
1. `capture_thread` — RTSP → `latest_frame[0]` (single-slot, drop-old). Signals `frame_event`.
2. `inference_thread` — wakes on `frame_event`, dedupes via `frame_seq[0]` (skips if seq unchanged), runs TFLite + optional crop passes + filter, picks max-confidence detection, computes new `(pitch, yaw)` setpoint via gain law, calls `link.update_setpoint(pitch, yaw)`.
3. `gimbal_tx_thread` (`GimbalLink.run_tx_loop`) — independent cadence. Transmits at `--send-rate` (default 20 Hz) when dirty bit is set, otherwise re-sends last setpoint at `--heartbeat-send-rate` (default 2 Hz). This keeps the autopilot/gimbal driver fed even when target is centered (no motion needed) or lost (HOLD).
4. `gimbal_rx_thread` (`GimbalLink.run_rx_loop`) — drains MAVLink inbox so the socket buffer can't fill. Logs `COMMAND_ACK` for cmd 205 and `STATUSTEXT` at any severity; everything else is silently consumed.

Main thread is the optional video writer (skipped under `--no-output`).

### Control mapping (pixels → MAVLink) — exact chain documented in this session
1. Top-confidence detection → bbox center `(cx, cy)` in OpenCV pixel coords (origin top-left, +x right, +y down).
2. Frame center `(fx, fy) = (W/2, H/2)`. Normalized error `err_x = (cx-fx)/fx`, `err_y = (cy-fy)/fy`. Range `[-1, +1]`, FPS+resolution independent.
3. Deadband: `|err_x| < 0.08 AND |err_y| < 0.08` → `direction_label="CENTERED"` → **no setpoint update** (this is how the loop settles).
4. Outside deadband:
   - `current_yaw   += yaw_gain * err_x` (target right → err_x +, yaw clockwise +)
   - `current_pitch -= pitch_gain * err_y` (target down → err_y +, pitch -, because image +y is down while gimbal +pitch is up)
5. Clamp to `[-90, +90]` yaw, `[-45, +45]` pitch.
6. `link.update_setpoint(pitch, yaw)` — sets dirty bit, tx thread picks it up.
7. tx thread emits `MAV_CMD_DO_MOUNT_CONTROL(205)` with `param1=pitch, param3=yaw, param7=MAVLINK_TARGETING(2)`.

### `--center-gimbal` mode (added mid-session)
Skips camera/TFLite/threading entirely. Connects MAVLink → calls `center_gimbal(master, duration, rate, dry_run)` → exits. Sends `(0, 0)` `--center-duration` × `--center-rate` times (defaults 2.0s × 5 Hz = 10 packets) because single-packet drops are common on the link and the gimbal driver expects sustained traffic to settle on a new setpoint. Honors `--no-mavlink` for bench dry-run.

### What does NOT work yet (intentional gaps, user-flagged)
- **Gains are not FOV-calibrated.** `yaw_gain=12` and `pitch_gain=10` are heuristic. A target at `err_x=1.0` (right edge) commands +12° yaw, but the actual angular offset of that pixel depends on the camera's horizontal FOV. Deadbeat tracking would require `yaw_gain ≈ HFOV/2` deg.
- **No gimbal feedback closed-loop.** `run_rx_loop` reads `MOUNT_STATUS` and `GIMBAL_DEVICE_ATTITUDE_STATUS` if present but doesn't compare commanded vs measured.
- **Body frame, not earth frame.** No roll/pitch compensation from airframe ATTITUDE.
- **No PWM fallback.** User asked about PWM-based centering mid-session, then redirected to mapping discussion before answering channel/neutral questions. Option remains open.

### Files this session
- New: `test/tf_live_inferenceV2_gimbal_auto.py` (~530 lines)
- New: `ailearn.sh` at repo root (copied from `~/Documents/aerospace2025-26/UAS_Competition_task_1_2026/CODEX/ailearn.sh`; was missing from this repo)
- Regenerated: `CODEX/AILEARN_REPORT.md`
