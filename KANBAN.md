# Kanban

## Done (2026-05-14 late #2 — fire path refactored: DO_REPEAT_RELAY + COMMAND_ACK-gated; NEEDS PI RE-VALIDATION)

After the DO_SET_RELAY + RELAY_STATUS-gated state machine was user-validated working, refactored the fire path on user request to use `DO_REPEAT_RELAY` (mirroring QGC's "Shoot Gun" exactly) with `COMMAND_ACK` as the confirmation source instead of `RELAY_STATUS`.

### Why the switch
- User: *"can we switch it to the do repeat relay with the same arming and disarming logic?"*
- First try used `RELAY_STATUS` to gate ARMING — script got stuck in ARMING because ArduPilot's `RELAY_STATUS` doesn't reliably show intermediate ON during a 1-cycle `DO_REPEAT_RELAY` (cycle ends where it started, so commanded final state = initial state).
- User identified the fix: *"we should also be able to check the ack command from do repeat relay to verify the states no?"*
- `COMMAND_ACK` for `cmd=182` arrives within ~100ms regardless of relay timing → reliable gate.

### Final architecture (4-phase, ACK-gated)
- `IDLE → ARMING → FIRING → COOLDOWN → IDLE`
- IDLE→ARMING: send ONE `DO_REPEAT_RELAY(relay=1, cycles=1, period=2*fire_period)`, mark `fire_send_time`.
- ARMING→FIRING: `COMMAND_ACK cmd=182 result=0` received after `fire_send_time` (or 1.0 s timeout → log warning, proceed).
- ARMING→IDLE: `COMMAND_ACK cmd=182 result≠0` (rejected — no cycle was started, no cooldown needed).
- FIRING→COOLDOWN: `now - fire_send_time >= fire_period` (timer).
- COOLDOWN→IDLE: `now - fire_send_time >= 2*fire_period + fire_cooldown` (full autopilot cycle + idle gap).
- BLANK mode (without `--live-fire`) auto-confirms ARMING after 0.1 s simulated lag.

### MAVLink traffic per fire
- Outbound: 1× `DO_REPEAT_RELAY`. ZERO keepalive, zero reassertion.
- Inbound: 1× `COMMAND_ACK`. Plus the always-on `RELAY_STATUS` at 5 Hz (visibility only).

### Implementation changes
- New: `GimbalLink._last_repeat_relay_ack_result`, `_last_repeat_relay_ack_time`.
- New: `GimbalLink.get_last_repeat_relay_ack() → (result, timestamp_or_None)`.
- `run_rx_loop` now captures the ACK for `cmd=182` (already logs it; now stores it).
- `fire_advance(centered)` rewritten as 4-phase machine. Removed `fire_state["last_send"]`, `lock_fired`, `phase_start_time` (replaced by `fire_send_time` which serves all phase timing).
- CLI flag help text updated for `DO_REPEAT_RELAY` semantics (`fire_period` is the ON-half of a `2*fire_period` cycle; `fire_cooldown` is additional idle AFTER the full cycle completes; total between fires = `2*fire_period + fire_cooldown`).

### Status: NEEDS PI RE-VALIDATION
The previous iteration (DO_SET_RELAY-based) was user-validated. This refactor is compile-clean but Pi-untested. Next Pi run should watch for `[ACK-RELAY] cmd=182 result=0` to confirm the autopilot accepted the command. If `result≠0` or timeout occurs frequently, fall back to the DO_SET_RELAY version (one git commit back) which was confirmed working.

### Things removed in this iteration (preserved in git history if needed)
- 2 Hz `DO_SET_RELAY` keepalive in every phase.
- `RELAY_STATUS`-bit confirmation for ARMING/DISARMING transitions.
- `DISARMING` phase (no longer needed — autopilot auto-OFFs after cycle ON-half).
- `lock_fired` flag (timer-based COOLDOWN serves the same gate).

## Done (2026-05-14 late — slew-rate-limited tracking + state-machine firing, USER-VALIDATED on real hardware)

### Gimbal control law (final architecture — replaces the cumulative-P integrator)
- **Two-tier control via `GimbalLink`**:
  - Inference thread computes `wished_yaw = link.get_current() + yaw_gain * err_x` fresh every frame (NOT integrated). Reference is actual transmitted position.
  - TX thread (20 Hz) ramps the transmitted `current_yaw` toward `wished_yaw` at `max_slew_rate_yaw / 20` deg per tick. Hard rate limit.
- New `GimbalLink` methods: `set_wished()`, `get_current()`, `get_wished()`. `run_tx_loop` now does the rate-limited ramping and transmits each tick.
- **CLI flags added**: `--max-slew-rate-yaw 2.0`, `--max-slew-rate-pitch 1.5` (deg/sec). Conservative micro-stepping (0.10°/0.075° per tick at 20 Hz). Sim-equivalent `--yaw-gain 12 / --pitch-gain 10` retained.
- **`--start-from-current` flag**: at startup, reads `MOUNT_STATUS` / `GIMBAL_DEVICE_ATTITUDE_STATUS` via new `read_current_gimbal_position()` helper. Uses autopilot-reported gimbal orientation as the initial reference. Without the flag the script centers the gimbal first via existing `center_gimbal()`.
- **Confirmed**: user observed smooth convergence without overshoot at `2.0/1.5 deg/sec`; `--start-from-current` correctly read `pitch=+37.71 yaw=+1.13` from the user's gimbal and tracked from that reference.
- **Per-frame log** now shows both `cur=(y,p)` (transmitted, actual gimbal) and `wished=(y,p)` (controller intent).

### Fire control — phase state machine with RELAY_STATUS confirmation (final architecture)
- **Five phases**: `IDLE → ARMING → FIRING → DISARMING → COOLDOWN → IDLE`. Transitions ARMING→FIRING and DISARMING→COOLDOWN require `RELAY_STATUS` confirmation from autopilot. FIRING→DISARMING and COOLDOWN→IDLE are time-based.
- **2 Hz keepalive** in every phase: continuously reasserts the target relay state via `DO_SET_RELAY`. Idempotent. Single-packet loss recovers within ~500 ms.
- **`MAV_CMD_SET_MESSAGE_INTERVAL` (511)** sent on connect to subscribe `RELAY_STATUS` (msg 376) at 5 Hz. `GimbalLink` captures the latest `on` / `present` bitmasks in `run_rx_loop`; `link.get_relay_status()` exposes them to the fire state machine.
- **`fire_period` timer starts on confirmed ON**, not on send. `fire_cooldown` timer starts on confirmed OFF. So "5-second burst" means exactly 5 s of confirmed ON.
- **CLI flags**:
  - `--live-fire`: real `DO_SET_RELAY` commands. Default is BLANK (logs phase transitions, no real fire).
  - `--no-fire`: disables fire logic entirely.
  - `--fire-relay 1`: matches QGC "Shoot Gun" action's `param1=1`.
  - `--fire-period 5.0`: seconds ON per burst.
  - `--fire-cooldown 0.5`: seconds OFF between back-to-back bursts. Set to 0 for one-shot-per-lock.
- **Abort path**: target leaves CENTERED during ARMING → drop to DISARMING (turn relay off before it actually fired). FIRING is *not* aborted on target loss — burst runs to completion.
- **Exit safety**: `finally` block always sends `DO_SET_RELAY` OFF on Ctrl-C / crash / normal exit.
- **Confirmed**: full IDLE→ARMING→FIRING→DISARMING→COOLDOWN cycle worked end-to-end on the user's autopilot. User explicitly approved: *"that worked"*.

### Manual gimbal control script — `--live-fire` re-purposed as one-shot CLI action
- `python3 manual_gimbal_control.py --live-fire` now: connect → send ONE `MAV_CMD_DO_REPEAT_RELAY` (`param1=1 param2=1 param3=2`, matching QGC "Shoot Gun" byte-for-byte) → wait up to 2.5 s for `COMMAND_ACK` → log result code → exit.
- No panel, no keyboard handler, no concurrent threads. Single linear code path.
- Used as the **isolation harness** that proved the relay command works when issued from a non-interactive context — confirmed firing on a single CLI invocation.
- Interactive panel (without `--live-fire`) still works for gimbal positioning; `f` key now simulates fire in BLANK only.

### Things tried and abandoned this session (for posterity)
- Per-frame step clamp + lock-and-hold state machine — reverted. Treated symptoms not root cause.
- Tiny gains (`--yaw-gain 0.5 / --pitch-gain 0.4`) cumulative-P — too slow, still overshot.
- Settle-delay between corrections — "messed up processing" per user.
- `DO_REPEAT_RELAY` double-send — caused on/off/on/off chatter from overlapping autopilot state machines.
- `DO_SET_RELAY` ON+timer+OFF with blind 2 Hz keepalive (no RELAY_STATUS check) — worked but no truth confirmation.
- Closed-loop mismatch correction without phase gating — risked corrective commands interfering with in-progress autopilot cycles.

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
- Added manual gimbal control + live telemetry CLI (2026-05-13):
  - `test/manual_gimbal_control.py` (Python 3.9-compatible; designed to run on the Pi)
  - Continuous keypress control via cbreak terminal mode (`w/s` pitch, `a/d` yaw, `c` center, `+/-` step, `r` resend, `q` quit) — no Enter needed; OS auto-repeat gives continuous motion while held
  - Uses identical MAVLink command shape as `tf_live_infrence_gimbal_live.py` — `MAV_CMD_DO_MOUNT_CONTROL` (cmd 205) absolute pitch/yaw in `MAV_MOUNT_MODE_MAVLINK_TARGETING`
  - Clamps to ±90° yaw / ±45° pitch
  - Live telemetry panel (ANSI-refreshing, 10Hz) showing last-value + age for: `HEARTBEAT`, `ATTITUDE`, `SYS_STATUS` (V/I/%), `GLOBAL_POSITION_INT`, `VFR_HUD`, `GPS_RAW_INT`, `MOUNT_STATUS`, `GIMBAL_DEVICE_ATTITUDE_STATUS` (quat→euler), `RC_CHANNELS`, `COMMAND_ACK`, `STATUSTEXT`. Each row prefixed `+` (fresh <2s), `!` (stale), `X` (never seen).
  - Requests `MAV_DATA_STREAM_ALL` from autopilot at 10Hz on connect
  - 2Hz background resend of current setpoint so the autopilot can't time out
  - Verified: parses under 3.9 grammar, dry-run renders correctly, non-TTY fallback prints warning instead of crashing

## Done (2026-05-14 — autonomous gimbal automation, USER-VALIDATED)
- **`test/tf_live_inferenceV2_gimbal_auto.py` ships and works end-to-end.** Combines V2 threaded inference + simulation gain law + manual-gimbal MAVLink plumbing in one production script for the flight computer.
  - 4-daemon-thread architecture: capture / inference (+ setpoint update) / gimbal-tx / gimbal-rx
  - `GimbalLink` class owns connection, dirty-bit setpoint, send-rate vs heartbeat-send-rate cadence, COMMAND_ACK + STATUSTEXT logging
  - Inherits all V2 features: `--no-output`, dual crop passes, CLAHE, `filter_detections`, seq-dedup
  - Inherits all simulation control: deadband `0.08`, cumulative-P gains `yaw=12 / pitch=10` deg-per-frame at full-scale error, clamp `±90° / ±45°`, exact same per-frame log format
  - Inherits all manual-gimbal connection robustness: `wait_heartbeat`, `request_streams`, transmit thread with periodic heartbeat resend at 2 Hz so gimbal driver never times out
  - `--center-gimbal` early-exit mode (skips camera/model setup; sends `(0,0)` for `--center-duration` seconds at `--center-rate` Hz)
  - Honors `--no-mavlink` for bench dry-run
  - User confirmed: "that script works amazingly. it just needs some logic tuning and gain control on the movement and mapping, but other than that its great"
- **`ailearn.sh` installed at repo root** (was missing; copied canonical version from sibling project)

## Done (2026-05-14 — overshoot fix + lock-and-hold)
- **Per-frame motion cap** added to `tf_live_inferenceV2_gimbal_auto.py` control law. New flags `--max-yaw-step` (default 2.0°) and `--max-pitch-step` (default 1.5°) clamp `gain*err` so a large pixel error cannot issue an oversized angle command and overshoot. Below the saturation point (`|err_x| < 0.166` at default gains) response is still fully proportional; above it, the step is capped.
- **Lock-and-hold state machine** added with `--lock-frames` (default 5) and `--shoot-duration` (default 3.0 s). State transitions:
  - `TRACKING` → in-deadband counter increments each centered frame; at `lock_frames` consecutive in-deadband frames, → `LOCKED`.
  - `LOCKED` ("SHOOTING") → tracking continues **even inside deadband** (corrects wind/drift); `LOCK_LOST_DRIFT` if target leaves deadband; `SHOOT_DONE` after `shoot_duration` seconds → back to `TRACKING` (counter reset).
  - Lost target → `LOCK_LOST_NO_TARGET`, drop to `TRACKING`.
- Per-frame log line now carries `state=`, `lock=N/N` or `t=X.XX/Y.YY s`, and `step=(y±X.XX,p±X.XX)`. Overlay shows shooting state in green; cross-hair line + bbox circle turn green during `LOCKED`.

## In Progress (2026-05-14 late — post-state-machine-confirmation)
- **Tune slew rates against actual gimbal hardware** — defaults `--max-slew-rate-yaw 2.0 / --max-slew-rate-pitch 1.5 (deg/sec)` are conservative micro-stepping. Confirmed smooth without overshoot. Next: if smoothness is good, raise toward `5-10 deg/sec` for faster target re-acquisition. The gain itself (`yaw_gain=12 / pitch_gain=10`, same as simulation) does not need to change — slew rate is the right tuning knob now.
- **Tune `fire_period` and `fire_cooldown` for mission requirements** — default 5 s ON, 0.5 s OFF gives ~5 s fire / 0.5 s rest auto-repeat while centered. Adjust to match the operational requirement (suppressive bursts vs single-shot precision, etc.). Set `--fire-cooldown 0` for one-shot-per-lock semantics.
- **FOV-aware control mapping** — for deadbeat tracking, `yaw_gain ≈ HFOV/2`, `pitch_gain ≈ VFOV/2`. HFOV/VFOV of the actual camera still not captured. With slew rate limiting, this is now lower-priority — the rate limit makes overshoot impossible even if gain is wrong.
- **Gimbal closed-loop feedback (open-loop today)** — `MOUNT_STATUS` and `GIMBAL_DEVICE_ATTITUDE_STATUS` are ingested by `run_rx_loop` but only used at startup for `--start-from-current`. Could compare commanded `current_yaw` vs reported attitude during operation to detect mechanical lag or failure.
- **PWM fallback** — open from prior session. `MAV_CMD_DO_SET_SERVO` cmd 183 channels and neutral PWM values not yet captured. Low priority since `DO_MOUNT_CONTROL` works.
- **Architecture decision for live inference deployment on Pi** (2026-05-13 evening — still open)
  - Confirmed via `--no-output` test: the **video output stream** (raw 1080p over UDP via `rtpvrawpay`) is the choke point on the Pi when inference is running, NOT inference itself, NOT the camera, NOT the decoder. With `--no-output`, inference runs steady at ~7 fps with no multi-second gaps. With output enabled at 1080p/raw, capture rate collapses to ~1 fps causing 7-second stalls in detections.
  - Inference ceiling on Pi: ~140 ms/cycle = **7 fps max sustained** (60% of FullDataSetProd ops fall back to CPU per compile log: 132 on TPU / 211 on CPU). Hardware floor short of re-exporting at smaller `imgsz`.
  - Source camera confirmed delivering 30 fps clean to the Pi (standalone gst pipeline reads 30 fps from `rtsp://10.42.0.1:8554/front_high`).
  - **Decision needed**: how to relay video to laptop without choking inference. Three options:
    1. Skip the relay: run with `--no-output` for max inference throughput. Detection data goes to telemetry/MAVLink, not to a viewer.
    2. Drop output resolution + framerate: 720p15 + H.264 encoding in pipeline4 (`x264enc tune=zerolatency speed-preset=ultrafast`) brings TX bitrate from ~250 Mbit/s to ~6 Mbit/s.
    3. Two-machine split: Pi runs inference only, laptop pulls camera RTSP directly for preview.
- Drone test integration planning:
  - map simulation control outputs (`yaw_rate`, `vx`, `vz`) to real MAVLink commands
  - validate sign conventions and gain scaling on airframe/simulator
- Drone test integration planning:
  - map simulation control outputs (`yaw_rate`, `vx`, `vz`) to real MAVLink commands
  - validate sign conventions and gain scaling on airframe/simulator
  - preserve current simulation overlays/logs as debug parity harness

## Done (2026-05-13 Evening Session — EdgeTPU compile + Live Inference tuning)
- **EdgeTPU compile of FullDataSetProd** — non-trivial; required TF 2.15 sidecar venv + flatbuffer surgery (full recipe at `HANDOFF.md § 12a`):
  1. Direct `.deb` install of `edgetpu_compiler` 16.0 (apt repo doesn't ship `noble`). URL: `https://packages.cloud.google.com/apt/pool/coral-edgetpu-stable/edgetpu-compiler_16.0_amd64_3ccd3b6ea6298eaaae6aa045764b3184.deb`.
  2. TF 2.19 (this venv) emits `CONV_2D v6` + collapses one depthwise conv into a grouped CONV_2D — compiler rejects both. Built sidecar Python 3.11 venv via `uv` with TF 2.15 (`venv-tf215/`), re-did int8 conversion from the saved_model (`build/convert_int8_tf215.py` using 500-image calib npy in `build/calib_500x3x640x640_float32.npy`).
  3. TF 2.15 still emits one grouped CONV_2D (filter `(128,3,3,1)`, in_c=128). Wrote `build/surgery_grouped_to_depthwise.py` to flatbuffer-patch that op back into DEPTHWISE_CONV_2D + inline-downgrade CONV_2D op-code version 6→3.
  4. Compile succeeded: `export/FullDataSetProd_full_integer_quant_edgetpu.tflite` sha256 `3d599378240246a660d707839aee6506bfaa44b1e135a354fc13b04dfda0d3f3` (3.05 MiB). 132 ops on TPU, 211 on CPU — same TPU/CPU split shape as `project1_prod` (~28% on TPU).
  5. Pre-compile source preserved: `export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_dwfix_v3.tflite` sha256 `b62b0f55ef94619d26f9ec2ff1db2cffb54617301dea388c2ef19f593c9b7f5d`.
- **Pi deployment + live inference tuning** (`test/tf_live_inferenceV2.py` on Pi at `~/MLBuilder/test/`):
  - Compiled artifact scp'd to Pi as `~/FullDataSetProd_edgetpu.tflite`. Confirmed `[ALLOCATE] TPU active: True`, output shape `(1,5,8400)` raw head (intentional — matches FullDataSetProd's `nms=False` export). Detections in expected location, confidence 0.85–0.92 on target.
  - Diagnosed bursty/freezing inference behavior across many iterations:
    1. Found stale-frame re-processing bug: `frame_event.wait(timeout=0.1)` returned regardless of event state; inference reprocessed the same frame multiple times. Fixed via `frame_seq` counter + dedup (`seq == last_seen: continue`).
    2. Diagnosed "waves" of output as **dual-clock drift** between writer sleep clock and capture's irregular arrival timestamps — was wrong; turned out to be writer-side CPU saturation.
    3. Tried collapsing writer into inference thread (paced by frame arrival) — better but still bursty because source provides ~4 fps with H.264 B-frame buffering at `latency=0`. Added `rtpjitterbuffer latency=200`.
    4. Suspected wifi half-duplex contention; ruled out — source standalone delivered 30 fps clean.
    5. Investigated `v4l2h264dec` for hardware H.264 decode on Pi (Trixie/Pi 5 doesn't expose it; only `/dev/video19` rpivid metadata node, no `/dev/video10-12` codec nodes). Installed `gstreamer1.0-plugins-bad` to no avail. Software `avdec_h264` is the only available decoder, but it keeps up with 30 fps when alone.
    6. Restored decoupled 3-thread architecture (capture / inference / writer) with FPS=30 writer on the main loop. Output stream + inference both supposed to run independently — but at FPS=30 1080p raw the writer's `videoconvert + rtpvrawpay + UDP send` work starved the GStreamer scheduler, dropped capture rate to ~1 fps, caused 7-second inference gaps.
    7. Reduced to FPS=15 — still bursting at 1080p. Verified by `[WRITE] seq=` increments lagging far behind real time.
    8. Added `--no-output` flag. With output disabled: inference cadence is steady at ~140ms (7 fps), no gaps. CONFIRMED: writer is the bottleneck.
  - Tunings/scripts landed:
    - `test/tf_live_inferenceV2.py` (updated): seq-dedup in inference thread, decoupled writer loop in main thread, `--no-output` flag, conditional `writer.release()` in `finally`.
    - `test/tf_live_inferenceV2_backup.py`: untouched reference copy of pre-`--no-output` version.
    - `pipeline3`: `rtspsrc latency=200 ! rtpjitterbuffer latency=200 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=1 sync=false` (no `videorate` cap).
    - `pipeline4`: `appsrc is-live=true do-timestamp=true block=false max-bytes=20000000 format=time ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=I420 ! rtpvrawpay ! udpsink sync=false async=false` (added appsrc properties to prevent backpressure blocking).

## Done (2026-05-13 Session — FullDataSetProd train + export)
- New Label Studio export staged: `project-1-at-2026-05-13-06-40-8e81e090/` (3,487 images, 7× prior dataset, single class `Target`, 1920×1080 source).
- Scene-aware 80/20 train/val split via dhash perceptual-hash clustering with largest-first balanced bin-packing assignment:
  - `test/prepare_dataset_split.py` (Hamming threshold 5, seed 0)
  - Result: 2,790 train / 697 val (exact 80/20), 1,026 clusters (largest 990 → kept entirely in train to prevent scene leak)
- Wrote dataset yamls: `data.yaml`, `data_calib.yaml` (full 3,487), `data_calib_subset.yaml` (500-image OOM-safe subset; see below)
- Trained `FullDataSetProd` from `yolo11n.pt`: 40 epochs, imgsz=640, batch=20, AdamW lr=0.002, AMP, ~16 min on RTX 3070 Laptop. Best val: **P=0.956, R=0.85, mAP50=0.951, mAP50-95=0.719**.
- Trained weights promoted: `export/FullDataSetProd.pt` (sha256 `f213fdf4b4a30bf97ebc561a0148638a707b12c1015842053a925e0c3495e971`). Last epoch: `export/FullDataSetProd_last.pt` (`a464d96d...`).
- Diagnosed and fixed venv export pipeline:
  - `onnx_graphsurgeon 0.5.8` → `0.6.1` (was crashing on `onnx.helper.float32_to_bfloat16` removed in newer onnx)
  - Discovered Ultralytics ignores cwd in favor of `~/.config/Ultralytics/settings.json`'s `runs_dir` — training output landed at `~/Documents/MLBuilder/runs/detect/build/out/FullDataSetProd/` instead of `~/Documents/aerospace2025-26/MLBuilder/build/out/FullDataSetProd/`. Worked around by copying best.pt to canonical `export/` location.
- Diagnosed OOM kills during int8 calibration with full 3,487-image set:
  - Two yolo OOM kills at 13 GB RSS each (16 GB system, system-oomd terminated yolo + Chromium). Chrome/Chromium got killed by the same cgroup OOM event, hence "VS Code crashed."
  - Ultralytics materializes the entire calibration set as numpy float32: 3487 × 640 × 640 × 3 × 4 bytes ≈ 17 GB
  - Fix: 500-image random subset (`calib_subset_500.txt`, seeded via `shuf --random-source=<(yes 0)`) — comfortably exceeds Ultralytics' >300 recommendation, peak ~3 GB calibration RAM
- Exported all int8 + float TFLite variants via Ultralytics: see `export/FullDataSetProd_saved_model/`. Full hash + IO dtype/quantization table in CONTEXT 2026-05-13 addendum.

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
1. **Choose live-inference deployment mode** (the open decision in In Progress):
   - If detection-only is enough → run with `--no-output`, route detections to MAVLink/telemetry only. No code change needed.
   - If video preview is needed → either implement H.264 encoded output (pipeline4 changes below) OR have the laptop pull RTSP directly from the camera bypassing the Pi.
2. **H.264 output path** (if chosen): replace `rtpvrawpay` with `x264enc tune=zerolatency speed-preset=ultrafast bitrate=6000 ! rtph264pay config-interval=1 pt=96` in pipeline4. Drops TX bitrate ~40× (from 250 to 6 Mbit/s); also update laptop receiver caps from `encoding-name=RAW` to `encoding-name=H264` with `rtph264depay ! avdec_h264`.
3. **Re-export at smaller imgsz** (optional, for higher inference rate): if you need >7 fps inference, re-export FullDataSetProd at `imgsz=320`. Reuses [[project-edgetpu-compile-workaround]] recipe — same TF 2.15 sidecar + flatbuffer surgery. Expected: ~4× faster inference, ~25 fps theoretical ceiling.
4. **Decide on output-shape convention**: re-export with `nms=True` to restore the `(1, 300, 6)` deployment fingerprint, or accept `(1, 5, 8400)` and switch deployment-identity check to sha256 hash. (Re-export needs another ~30 min int8 calibration on 500-image subset.)
5. After deploy mode chosen, capture metrics from a real target session:
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
5. Pi production with FullDataSetProd (new):
   - **Inference-only (recommended, no choke)**: `python3 -B tf_live_inferenceV2.py ~/FullDataSetProd_edgetpu.tflite --tpu -p --no-output`
   - **With output stream (currently chokes — needs H.264 fix to pipeline4)**: `python3 -B tf_live_inferenceV2.py ~/FullDataSetProd_edgetpu.tflite --tpu -p -o`
   - **Fallback to project1_prod** if FullDataSetProd misbehaves: `python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o`
6. Verify Pi source feed standalone (diagnostic — confirms camera is 30 fps regardless of script):
   - On Pi: `/usr/bin/python3 test.py` with the frame-counter test script (see HANDOFF § 13)
7. Pi probe (TFLite metadata inspection):
   - file at `/tmp/probe.py` on Pi; uses `tflite_runtime` with EdgeTPU delegate.
7. Manual gimbal control + telemetry monitor (run on Pi after `scp test/manual_gimbal_control.py pi@10.42.0.1:~/`):
   - `python3 ~/manual_gimbal_control.py`                               (default `tcp:10.42.0.1:5760`)
   - `python3 ~/manual_gimbal_control.py --mavlink udpin:0.0.0.0:14550` (alt endpoint if autopilot is UDP)
   - `python3 ~/manual_gimbal_control.py --no-mavlink`                  (offline panel check, no transmit)
