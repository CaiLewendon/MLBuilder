# CLAUDE.md — MLBuilder Project Operating Notes

## ⏸ WHERE WE LEFT OFF (2026-05-17 — `FullDataSetProdV2` TRAINED + WEIGHTS PROMOTED; int8 export + EdgeTPU compile NOT YET DONE)

**Session summary:**
- User provided new Label Studio export: `downloadedUpdatedProductiondata/` (3727 images, single class "Target", ~7% larger than the prior 3487-image FullDataSetProd dataset). Same Label Studio "YOLO with Images" format.
- Ran `test/prepare_dataset_split.py --dataset downloadedUpdatedProductiondata --val-ratio 0.2 --hash-threshold 5 --seed 42` → **2982 train / 745 val**. 1027 clusters, 802 singletons, largest cluster 990 images (same near-stationary camera sequence as before — kept whole in train).
- GPU was inaccessible (kernel-vs-driver mismatch — duplicate `nvidia-driver-550` + `nvidia-driver-580` installed). User rebooted; driver 580.142 came back clean, RTX 3070 Laptop online, 7.4 GB free.
- Trained `yolo11n.pt` → `FullDataSetProdV2` for 40 epochs, batch=20, imgsz=640 — **same recipe as FullDataSetProd**. Wall time **0.282 hours (~17 min)**. AdamW auto-selected (lr=0.002, momentum=0.9). Ultralytics again ignored `project=build/out` and wrote to `~/Documents/MLBuilder/runs/detect/build/out/FullDataSetProdV2/`.

**Training results (val on best.pt):**
| Metric | FullDataSetProdV2 (new) | FullDataSetProd (prior, deployed) | Delta |
|---|---|---|---|
| Precision | 0.899 | 0.956 | -0.057 |
| Recall | 0.891 | 0.85 | **+0.041** |
| mAP50 | 0.947 | 0.951 | ~same |
| mAP50-95 | 0.646 | 0.719 | -0.073 |

Higher recall, slightly lower precision and bbox-localization quality. Net mAP50 essentially identical. Field validation needed to judge real-world effect.

**Artifacts on disk:**
- `export/FullDataSetProdV2.pt` (5.45 MB, sha256 `d7ad6f995bbf52d164caa88c612c17e93c2c4fae626f75d41a841915692894e0`)
- `export/FullDataSetProdV2_last.pt` (5.45 MB)
- `downloadedUpdatedProductiondata/train.txt` (2982 entries)
- `downloadedUpdatedProductiondata/val.txt` (745 entries)
- `downloadedUpdatedProductiondata/calib_all.txt` (3727 entries — for later subset)
- `downloadedUpdatedProductiondata/data.yaml` and `data_calib.yaml`
- `build_FullDataSetProdV2_train.log` (full training console output)

**REMAINING STEPS (next session — resume here):**

1. **Build 500-image calibration subset** (OOM-safe per [[int8-calibration-cap]]):
   ```bash
   cd downloadedUpdatedProductiondata && \
   shuf -n 500 --random-source=<(yes 42) calib_all.txt > calib_subset_500.txt
   # Write data_calib_subset.yaml referencing calib_subset_500.txt
   ```

2. **Export to int8 TFLite** (~35 min wall time per prior session):
   ```bash
   venv/bin/yolo export model=export/FullDataSetProdV2.pt format=tflite int8=True \
     data=downloadedUpdatedProductiondata/data_calib_subset.yaml imgsz=640
   ```
   Watch for: `onnx_graphsurgeon` version compatibility (last time required bump to 0.6.1); OOM kill from systemd-oomd if memory pressure (mitigated by the 500-image cap).

3. **EdgeTPU compile** via TF 2.15 sidecar + flatbuffer surgery — see [[edgetpu-compile-workaround]] memory. Required because TF 2.19 op-versions don't match the EdgeTPU compiler's compat range; the surgery converts grouped `CONV_2D` → `DEPTHWISE_CONV_2D` (version 6 → 3).

4. **Deploy to Pi**: scp the resulting `FullDataSetProdV2_edgetpu.tflite` to `~/FullDataSetProdV2_edgetpu.tflite`. Verify hash post-compile. Confirm `[OUT] shape=(1, 5, 8400)` on live inference (raw-head, same as FullDataSetProd).

5. **A/B compare on the Pi**: run the same scene against both models with `--sharpen 0.4` and compare per-frame confidence + false-positive rate. Decide whether to keep V2 or stay on V1.

**Key context for resume:**
- `--sharpen 0.4` is the production preproc recipe — applies identically to V2.
- `tf_live_inferenceV2_final_auto.py` is model-agnostic and will run V2 by swapping the model path; no script changes needed.
- The 990-image scene cluster is the dominant feature in train.txt; val (745 images) is "all other scenes" and a pessimistic benchmark.

**To resume:** re-invoke me and say *"continue from the int8 export"*. I'll pick up at Task #12 (build calib subset) and auto-progress through 13 (export) + 14 (TPU compile).

**Earlier 2026-05-16 session (final_auto build) — still applies: `tf_live_inferenceV2_final_auto.py` is ready for first Pi BLANK-mode run with whatever model is currently deployed (FullDataSetProd until V2 ships).**

---

## ⏸ PRIOR WHERE WE LEFT OFF (2026-05-16 evening — `tf_live_inferenceV2_final_auto.py` BUILT for Big City RPAS Task 2; compile-clean, READY FOR FIRST PI BLANK-MODE RUN)

**Session summary:**
- Built `test/tf_live_inferenceV2_final_auto.py` (2821 lines) — combines drone_auto + gimbal_auto into a single sequential one-shot **Task 2 Fire Extinguishing** engagement script.
- **Mission flow** (owned by inference thread, single source of truth):
  `PHASE_DRONE_POSITIONING → PHASE_HANDOFF_WAIT → PHASE_GIMBAL_TRACKING → PHASE_FIRING → PHASE_VERIFY → PHASE_HANDBACK → PHASE_DONE`.
  - DRONE_POSITIONING: drone_auto's 7-state machine drives; gimbal held STATIC at startup angle.
  - HANDOFF_WAIT: count `--handoff-confirm-frames` (default 15) consecutive FINAL_HOLD frames; reset to 0 on regression.
  - GIMBAL_TRACKING: drone enters FREEZE (continuous zero-velocity SET_POSITION_TARGET_LOCAL_NED at `--tx-rate` to maintain hover); gimbal slew-rate-limited tracking activates; discharge state machine ticks.
  - FIRING: 5-second water discharge (DO_REPEAT_RELAY, COMMAND_ACK-gated, same primitive as the gun trigger). Drone stays frozen.
  - VERIFY: capture 5 frames over 2 s, save best (highest conf, Laplacian-variance fallback) as `Task_2_<team_name>_target_<#>_<ts>.jpg`. Print operator declaration warning.
  - HANDBACK: `master.set_mode_apm(args.handback_mode)` → LOITER (default), poll for ACK or timeout 2.0 s, send_halt + defensive DO_SET_RELAY OFF, exit.
- **Single MAVLink master** shared by `MovementLink` and `GimbalLink` via a new `shared_rx_loop(master, drone_link, gimbal_link, on_handback_mode_ack, stop_event)` that dispatches by msg type. Each Link exposes a `handle_message(msg)` extracted from its old `run_rx_loop` body. Justification: pymavlink's `recv_match` is not safe to race across threads on one socket — both source scripts already had one rx thread; we merged them.
- **6 threads**: main + capture + inference + drone_tx + gimbal_tx + shared_rx. Spawn order: rx → drone_tx → gimbal_tx → cap → inf (rx first so initial telemetry is captured before tx loops gate on `is_guided()`).
- **Task 2 compliance gate**: `--min-start-distance-cm 200` aborts with exit code 7 if the first DISTANCE_SENSOR reading is < 2 m (Task 2 §5.2.4 requires the autonomous approach to start from >2 m).
- **CLI namespace resolution**: drone's `--deadband` / `--yaw-gain` renamed to `--drone-deadband` / `--drone-yaw-gain`; gimbal's renamed to `--gimbal-deadband` / `--gimbal-yaw-gain`; gimbal's `--start-from-current` → `--start-from-current-gimbal`; gimbal's one-shot `--center-gimbal` mode → `--center-gimbal-at-start` (mutex with start-from-current; default centers). New: `--handoff-confirm-frames`, `--handback-mode {LOITER,RTL,ALT_HOLD,LAND}`, `--handback-mode-timeout`, `--min-start-distance-cm`, `--team-name`, `--target-number`, `--photo-output-dir`, `--no-photo-capture`, `--capture-frame-count`, `--capture-frame-interval`.
- **Compile + CLI verified**: `python3 -m py_compile` clean; `--help` exposes all flags; validators reject `--live-fly + --no-mavlink`, `--tx-rate < 4.0`, `--no-fire + --live-fire`, invalid `--handback-mode` choice, `--handoff-confirm-frames 0`.
- **NOT YET RUN** end-to-end. First test is Pi BLANK-mode with autopilot connected + drone disarmed in GUIDED.

**Pi run commands:**
```bash
# Pi BLANK-mode test (autopilot connected, drone disarmed in GUIDED; safest first run)
python3 -B tf_live_inferenceV2_final_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --sharpen 0.4 \
  --start-from-current-gimbal --team-name dev_test --target-number 1

# Pi LIVE engagement (observer on RC, GUIDED + armed, drone >2 m from target)
python3 -B tf_live_inferenceV2_final_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --sharpen 0.4 \
  --start-from-current-gimbal --live-fly --live-fire \
  --team-name <team> --target-number 1 \
  --max-vx 0.20 --max-vz 0.15 --max-yaw-rate 10.0 \
  --target-distance-cm 300 --distance-tolerance-cm 30

# Laptop dry-run (no MAVLink, simulated LiDAR; --no-photo-capture skips PHASE_VERIFY)
venv/bin/python test/tf_live_inferenceV2_final_auto.py \
  export/project1_prod_saved_model/project1_prod_float16.tflite \
  -p --video 0 --no-mavlink --overlay \
  --team-name dev_test --target-number 1 --no-photo-capture
```

**What to watch for at each phase transition (BLANK-mode logs):**
```
[MISSION t=  0.00s] DRONE_POSITIONING -> DRONE_POSITIONING | reason=...   # initial
[COMPLIANCE OK] start distance Xcm >= 200cm (Task 2 >2m criterion satisfied)
[STATE   X.XXs] NO_TARGET -> CENTERING ...
... drone state machine cycles ...
[STATE   X.XXs] ALTITUDE_ADJUST -> FINAL_HOLD ...
[MISSION t= X.XXs] DRONE_POSITIONING -> HANDOFF_WAIT | reason=drone reached FINAL_HOLD
[MISSION t= X.XXs] HANDOFF_WAIT -> GIMBAL_TRACKING | reason=FINAL_HOLD stable for 15 frames
[BLANK ARMING] sent DO_REPEAT_RELAY(1,cycles=1,period=10.00s); awaiting COMMAND_ACK for cmd=182
[BLANK ARMED] BLANK simulated ACK after 0.10s; ...
[MISSION t= X.XXs] GIMBAL_TRACKING -> FIRING | reason=discharge state machine entered FIRING
[BLANK BURST DONE] 5.00s ON elapsed; ...
[MISSION t= X.XXs] FIRING -> VERIFY | reason=discharge complete (5.0s); capturing photos
[VERIFY] captured frame 1/5 (conf=0.812); ...
[VERIFY] captured frame 5/5 ...
[VERIFY DECLARED] Photo saved at extinguish_photos/Task_2_dev_test_target_1_<ts>.jpg
[MISSION t= X.XXs] VERIFY -> HANDBACK | reason=photo captured + saved
[HANDBACK BLANK] would request mode=LOITER
[MISSION t= X.XXs] HANDBACK -> DONE | reason=handback timeout / BLANK
[MISSION DONE] team=dev_test target=1
```

**Next-session priorities (in order):**
1. **Pi BLANK-mode run of `tf_live_inferenceV2_final_auto.py`** — autopilot connected, drone DISARMED in GUIDED. Verify full phase progression + photo save + LOITER handback request (will be BLANK log only since no `--live-fly`).
2. **Mode-loss recovery test** — switch out of GUIDED mid-run; verify `[MODE LOST]` log rate-limited to once per 2 s; switch back to GUIDED, sends resume.
3. **Conservative LIVE engagement** — observer on RC override; drone >2 m from target; `--live-fly --live-fire` with capped `--max-vx 0.20 --max-vz 0.15`. Confirm LOITER ACK arrives and pilot regains manual control.
4. **Model retraining for Task 2 targets** — `FullDataSetProd` is trained on white plate; Task 2 targets are purple/blue paper circles 5-30 cm diameter on white backing. Without retraining, detection confidence on the actual competition targets is unverified.
5. **Field test against Task 2 mockup** — purple paper circle target with cabbage-juice dye; operator flies into search volume manually, engages GUIDED, then runs the script.

**Explicitly out of scope of `final_auto.py`** (separate concerns):
- GPS waypoint navigation to the building (operator flies manually into search volume per Big City RTM SOPs).
- Multi-target search across the unknown-count search volume (script handles ONE target per invocation; operator increments `--target-number` and re-runs).
- Indoor target navigation through the 3.5 m × 3 m doorway.
- Automatic Google Drive upload (script saves locally; manual upload preserves operator final-confirmation before declaration).
- Post-extinguish color verification (purple → blue CV check on the bbox) — would reduce false-declaration risk; future enhancement.

**Earlier 2026-05-16 work (preprocessing flags, --sharpen 0.4) is now baked into final_auto as well — all preprocessing flags carried forward identically. See [[preproc-sharpen-winner]], [[preproc-no-stacking]], [[lighting-dominates-conf]].**

---

## ⏸ PRIOR WHERE WE LEFT OFF (2026-05-16 — preprocessing flags landed across all V2 scripts; `--sharpen 0.4` validated as production recipe)

**Session summary:**
- Diagnosed a detection regression on the Pi: confidence collapsed from prior 0.85-0.92 down to 0.05-0.30 on the same scene + same model (hash verified). Source RTSP feed confirmed healthy (30 fps clean, zero drops, ~4.3 Mbit/s).
- Root cause: **lighting / local contrast on the target.** Pointing a flashlight at the target (with no visible change to the human eye) restored confidence to 0.85. Mechanism: tiny logit shifts produce huge sigmoid-probability swings on a ~38 px letterboxed target.
- Added six preprocessing flags to `test/tf_live_inferenceV2.py` (and ported identically to `tf_live_inferenceV2_gimbal_auto.py` and `tf_live_inferenceV2_drone_auto.py`): `--grayscale`, `--luminance`, `--contrast`, `--saturation`, `--sharpen`, `--sharpen-sigma`. All default to no-op so existing behavior is preserved when no flags are passed. Startup banner `[PREPROC] ...` reports the active set.
- **Sweep result — production recipe is `--sharpen 0.4` alone** (sigma=1.0 default). Consistent 0.74-0.80 confidence. See `[[preproc-sharpen-winner]]` memory.
- Non-obvious finding: **stacking CLAHE + sharpen is WORSE than either alone** (0.08-0.26 vs 0.59-0.74). Heavy combos (grayscale + saturation 0.6 + sharpen 0.7 + clahe) break detection entirely by shifting input too far from training distribution. See `[[preproc-no-stacking]]`.
- Source camera framerate is now 60 fps (was documented as 30 fps); H.264-encoded ~4.3 Mbit/s. No issue for inference but worth noting.

**Production commands (current):**
```bash
# Pi production inference (laptop-equivalent confidence on dim/varied lighting)
python -B tf_live_inferenceV2.py ~/FullDataSetProd_edgetpu.tflite --tpu -p --no-output \
  -l ../target_detector_labels.txt --sharpen 0.4

# Autonomous gimbal + fire (production recipe applied)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --start-from-current --live-fire \
  --sharpen 0.4

# Autonomous drone-movement BLANK-mode Pi test
python3 -B tf_live_inferenceV2_drone_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --sharpen 0.4
```

**Next-session priorities (unchanged from 2026-05-15, plus this session's adds):**
1. Pi BLANK-mode test of `tf_live_inferenceV2_drone_auto.py` (autopilot DISARMED, GUIDED mode).
2. Pi re-validation of the DO_REPEAT_RELAY + COMMAND_ACK gimbal-fire path.
3. Decide on durable lighting fix: add fixed task lighting to the demo rig vs. retrain FullDataSetProd with varied-lighting augmentation.
4. Field test combined: drone_auto + gimbal_auto as two processes.

---

## ⏸ PRIOR WHERE WE LEFT OFF (2026-05-15 — drone movement script built; NEEDS Pi BLANK-mode + flight test)

**New artifact this session:** `test/tf_live_inferenceV2_drone_auto.py` (1727 lines). Production drone-positioning script that issues real MAVLink `SET_POSITION_TARGET_LOCAL_NED` velocity setpoints in body frame to an ArduPilot Copter in GUIDED mode. Combines:
- 4-thread architecture from `tf_live_inferenceV2_gimbal_auto.py` (capture / inference / drone_tx / drone_rx)
- 7-state control machine from `tf_live_infrence_drone_simulation.py` (NO_TARGET → CENTERING → APPROACH → HOLD → LOCKED_HOLD → ALTITUDE_ADJUST → FINAL_HOLD) — ported VERBATIM with same gains, deadbands, confirmation-frame logic, sticky altitude target after lock
- New `MovementLink` class mirroring `GimbalLink` pattern: lock-protected state, run_tx_loop/run_rx_loop, but commands `SET_POSITION_TARGET_LOCAL_NED` (msg 84) instead of `DO_MOUNT_CONTROL`
- `--live-fly` flag (DEFAULT OFF = BLANK mode logs what it would send; on = real sends)
- Startup GUIDED-mode gate via `master.flightmode` check; exits with code 4 if not GUIDED. NO auto-switch, NO mode-change spam during run. Mode change during operation logged via HEARTBEAT monitoring; tx loop suppresses sends when not GUIDED.
- `DISTANCE_SENSOR` (msg 132) subscribed at 5 Hz for real range-to-target (replaces simulated LiDAR). Falls back to simulation when `--simulate-distance` or `--no-mavlink`.
- Telemetry feedback via `VFR_HUD` (groundspeed/climb shown as `feedback=` line in OSD beside the commanded velocity)
- Exit safety: sends one final zero-velocity `SET_POSITION_TARGET_LOCAL_NED` to halt the drone before disconnect.

**Status:** compile-clean, CLI validation passes (--live-fly+--no-mavlink conflict catches, --tx-rate<4 rejects, etc.). **Not yet tested on the Pi.** Next session priority: BLANK-mode Pi test with autopilot DISARMED to validate state machine + telemetry, then conservative `--live-fly` hover test with observer on the kill switch.

**Run commands (Pi):**
```bash
# BLANK mode (safe, no real velocity commands sent)
python3 -B tf_live_inferenceV2_drone_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760

# LIVE-FLY (real flight — observer on RC override at all times)
python3 -B tf_live_inferenceV2_drone_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --live-fly \
  --max-vx 0.20 --max-vz 0.15 --max-yaw-rate 10.0 \
  --target-distance-cm 300 --distance-tolerance-cm 30

# Pure dry-run (no autopilot at all — uses simulated LiDAR like drone_simulation)
python3 -B tf_live_inferenceV2_drone_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --no-mavlink
```

The existing gimbal+fire script (`tf_live_inferenceV2_gimbal_auto.py`) is unchanged — it remains the gimbal aiming + autonomous firing path. The new drone script is movement-only; if both are needed in flight, run them as two processes (each opens its own MAVLink connection to `tcp:10.42.0.1:5760`).

See `[[drone-automation-script]]` memory for full architecture details and per-phase behavior.

---

## ⏸ PRIOR WHERE WE LEFT OFF (2026-05-14 late — autonomous tracking + ACK-gated DO_REPEAT_RELAY firing)

**Two subsystems live in `test/tf_live_inferenceV2_gimbal_auto.py`. Tracking has been user-validated end-to-end. Fire path was just refactored from the DO_SET_RELAY ON/OFF + RELAY_STATUS-gated design (user-validated earlier in the session) to a simpler DO_REPEAT_RELAY + COMMAND_ACK-gated design — needs a Pi test to re-validate before being called "final".**

### 1. Gimbal control law — slew-rate-limited proportional (USER-VALIDATED, final)
The integrator runaway diagnosed earlier in the day was fixed by **decoupling the controller from the transmitted setpoint via `GimbalLink`**:
- Inference thread computes `wished_yaw = link.get_current() + yaw_gain * err_x` **fresh every frame** (NOT integrated). The reference is the actual transmitted position, not a free-running counter.
- TX thread (20 Hz) ramps `current_yaw` toward `wished_yaw` by at most `max_slew_rate_yaw / 20` degrees per tick. Hard rate-limit. Gimbal can never be commanded faster than its physical slew can keep up with.
- Per-frame log carries both `cur=(y,p)` (transmitted, what gimbal sees) and `wished=(y,p)` (controller intent, leads by `gain * err`).
- Defaults: `--yaw-gain 12`, `--pitch-gain 10` (sim values, unchanged), `--max-slew-rate-yaw 2.0`, `--max-slew-rate-pitch 1.5` (deg/sec).
- New `--start-from-current` flag: reads `MOUNT_STATUS` / `GIMBAL_DEVICE_ATTITUDE_STATUS` via `read_current_gimbal_position()` and starts the controller from the gimbal's actual orientation. Without the flag, the script centers the gimbal first. **Confirmed working** — user's gimbal reported `pitch=+37.71 yaw=+1.13` and tracking continued smoothly from that reference.

### 2. Fire control — DO_REPEAT_RELAY + COMMAND_ACK-gated state machine (CURRENT — needs Pi re-test)
After validating the DO_SET_RELAY ON+timer+OFF + RELAY_STATUS-gated design with the user, we refactored to use `DO_REPEAT_RELAY` (mirroring the QGC "Shoot Gun" command shape) with `COMMAND_ACK` as the gate instead of `RELAY_STATUS`.
- **4-phase state machine**: `IDLE → ARMING → FIRING → COOLDOWN → IDLE`.
- **IDLE→ARMING**: send ONE `DO_REPEAT_RELAY(relay=1, cycles=1, period=2*fire_period)` and mark `fire_send_time`. ArduPilot's cycle does `period/2` seconds ON then `period/2` seconds OFF; with `period = 2*fire_period`, the ON-half equals `fire_period`.
- **ARMING→FIRING**: gated on `COMMAND_ACK` for cmd=182 received AFTER `fire_send_time`.
  - `result=0` (ACCEPTED) → FIRING (autopilot confirmed it's running the cycle)
  - `result≠0` (DENIED/FAILED/UNSUPPORTED) → IDLE, log error, NO cycle was started so no cooldown needed
  - Timeout 1.0 s with no ACK → FIRING with WARN log (assume ACK was lost; honor the cycle timing in COOLDOWN)
- **FIRING→COOLDOWN**: timer-based at `now - fire_send_time >= fire_period`. Autopilot owns the actual relay timing.
- **COOLDOWN→IDLE**: timer-based at `now - fire_send_time >= 2*fire_period + fire_cooldown`. Waits for full autopilot cycle + cooldown gap so the next DO_REPEAT_RELAY can't collide with an in-progress one.
- **`RELAY_STATUS` (msg 376)** is subscribed at 5 Hz via `SET_MESSAGE_INTERVAL` and read for log visibility (`RELAY[1]=ON/OFF/?` in burst-done logs) but **does NOT gate transitions** — ArduPilot's RELAY_STATUS may not show intermediate ON during a 1-cycle DO_REPEAT_RELAY (ends where it started), which is why RELAY_STATUS-gating got us stuck in ARMING with this command.
- **Exactly ONE `DO_REPEAT_RELAY` per fire cycle**. No keepalive, no reassertion. Zero command spam. The `COMMAND_ACK` is inbound from autopilot.
- BLANK mode (`--live-fire` absent) auto-confirms ARMING after 0.1 s simulated lag.

### Key learnings from the long debugging arc (don't repeat)
- **`MAV_CMD_DO_REPEAT_RELAY` (182) with `cycles=1, period=N`** does **N/2 seconds ON, N/2 seconds OFF** in ArduPilot. So `--fire-period 5` requires `param3 = 2*fire_period = 10`. NOT "ON for N seconds" directly — the script computes the cycle_time internally.
- **Double-sending `DO_REPEAT_RELAY` (50 ms apart)** caused on/off/on/off chatter — each command starts its own relay-pulse state machine in the autopilot, and concurrent state machines collide. **Single-send only** for that command.
- **Sending `DO_SET_RELAY` OFF while a `DO_REPEAT_RELAY` cycle is in progress** interrupts the cycle. Mixing commands is brittle UNLESS the interrupt is intentional (abort / safety / exit cleanup).
- **`RELAY_STATUS` is unreliable for gating DO_REPEAT_RELAY phase transitions** because it reports the commanded *final* state, and a 1-cycle DO_REPEAT_RELAY ends where it started. **Use `COMMAND_ACK` instead** — it confirms the autopilot accepted the command within ~100 ms regardless of relay timing.
- **The right primitive choice**: `DO_SET_RELAY` (181) when the SCRIPT owns the on-time and you want fine control + intermediate state visibility. `DO_REPEAT_RELAY` (182) when the AUTOPILOT owns the on-time and you want to mirror a QGC button exactly. Pick one and stick with it.
- **QGroundControl's "Shoot Gun" action** is `cmd=182 param1=1 param2=1 param3=2` — relay 1, 1 cycle of 2 s = 1 s pulse. Verified by user. Our autonomous script sends the same command shape but with `param3 = 2*fire_period`.
- **The user's autopilot publishes `RELAY_STATUS`** AND ACKs `cmd=182` correctly — both confirmed at runtime.

### Working artifacts on disk
- `test/tf_live_inferenceV2_gimbal_auto.py` — autonomous tracking (validated) + ACK-gated DO_REPEAT_RELAY firing (needs Pi re-test).
- `test/manual_gimbal_control.py` — `--live-fire` is a one-shot fire-and-exit (parallel to `--center-gimbal`). Press `f` in panel mode for BLANK simulation; pass `--live-fire` as a CLI arg for real one-shot fire. Used as the isolation harness that confirmed the autopilot's relay control.

### Pi commands (current production)
```bash
# Autonomous tracking + autonomous firing (live)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --start-from-current --live-fire

# Autonomous tracking only (no fire commands sent — BLANK simulation in logs)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --start-from-current

# Autonomous tracking, no fire logic at all (camera/gimbal only)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
  --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --start-from-current --no-fire

# Manual one-shot fire test (NO panel, NO key handler)
python3 ~/manual_gimbal_control.py --live-fire

# Manual gimbal panel (BLANK fire simulation only — 'f' key logs but doesn't fire)
python3 ~/manual_gimbal_control.py
```

### Tuning knobs (defaults work as starting points)
- `--max-slew-rate-yaw 2.0 --max-slew-rate-pitch 1.5` (deg/sec) — controls gimbal slew. Lower = smoother but slower convergence; raise to 5-10 for faster tracking if the physical gimbal can keep up.
- `--fire-period 5.0` — seconds the relay is held ON per burst.
- `--fire-cooldown 0.5` — seconds of forced OFF between back-to-back bursts (set to 0 to disable auto-repeat — one-shot per centering event, requires un-center to re-arm).
- `--fire-relay 1` — relay instance (matches QGC config).

Output shape `(1, 5, 8400)` is INTENTIONAL — see `feedback_hash_not_shape` memory before assuming the wrong model is loaded.

---

## Read these first (in this order)
1. This file (you're here).
2. `CONTEXT.md` — deployment topology, model artifacts, latest session state.
3. `CODEX.md` — chronological session log of fixes, diagnoses, and decisions.
4. `KANBAN.md` — current tasks and blockers.
5. `HANDOFF.md` — runbook + decision trees.

## Project at a glance
Single-class "Target" detection. Camera → Raspberry Pi (Coral EdgeTPU inference) → Relay → Laptop viewer. Production script `tf_live_inferenceV2.py` runs on the Pi (NOT in the repo). Wrapper code in `MLBuilder/model/tflite/tflitemodel.py` handles TFLite loading, quantization, postprocessing.

## CRITICAL: Two distinct int8 EdgeTPU models exist

| Hash prefix | Build | Output shape | Status |
|---|---|---|---|
| `e4623d5d` | March 13 (legacy) | `(1, 5, 8400)` raw head | DEPLOYED on Pi (WRONG — pre-`project1_prod` training) |
| `153b25f3` | April / May (`project1_prod`) | `(1, 300, 6)` postprocessed | CORRECT (already on Pi at `~/target_detector_int8_edgetpu.tflite`, just not the default) |

Canonical "best" file per user: `export/project1_prod_int8_edgetpu_compat.tflite` (hash `153b25f3...`). Byte-identical to `target_detector_int8_edgetpu.tflite`, `export/project1_prod_full_integer_quant_edgetpu.tflite`, and `export/pi_rebuild_edgetpu/target_detector_int8_edgetpu.tflite`.

## MANDATORY first step on any TFLite/EdgeTPU debugging

```bash
# On the Pi
sha256sum ~/model_full_integer_quant_edgetpu.tflite
```

Match against the table above. If hash is `e4623d5d...`, the wrong model is loaded — fix that BEFORE investigating code, quantization, calibration, or training data. A whole session was burned on 2026-05-12 because this check was skipped.

## Visual signature of correctly-deployed model in live logs
- `[OUT] shape=(1, 300, 6)` → correct (`project1_prod`)
- `[OUT] shape=(1, 5, 8400)` → wrong (old March model)

## Things verified correct and not to be modified reflexively
- `MLBuilder/model/tflite/tflitemodel.py` — handles both architectures, reads quant params from the model file dynamically. The `[N,6]` branch gate `6 <= shape[1] <= 16 AND shape[0] > shape[1]` is invariant — do not loosen.
- NMS path uses `xywh` format throughout (fixed 2026-05-07/08).
- Input preprocessing: `q = round((pixel/255)/scale + zero)`, clip, cast. Correct for any int8 input model.
- Output dequantization: `(raw - zero) * scale` when int8/uint8 and scale > 0.
- Diagnostic prints `[TFLITEMODEL] Loaded from:` and `[ALLOCATE] ...` — keep these.

## Things ruled out as bug causes (don't chase again)
- Input/output quantization math in the wrapper.
- Branch gate selection between raw-head and postprocessed.
- The `calibration_image_sample_data_20x128x128x3_float32.npy` file at repo root — never used by deployed builds (real `data.yaml`-based calibration confirmed via build log).
- Video pipeline resolution (1080p / 640×360) — wrapper letterboxes to 640×640 internally.

## Production command (after model swap on Pi)
```bash
python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu \
  -l ../target_detector_labels.txt -p -o
```

## Laptop simulation commands (known-good)
```bash
venv/bin/python test/tf_live_infrence_gimbal_simulation.py --video 0
venv/bin/python test/tf_live_infrence_drone_simulation.py --video 0
```

## Memory system
Per-session learnings are stored at:
`/home/caile/.claude/projects/-home-caile-Documents-aerospace2025-26-MLBuilder/memory/`

Highlights:
- `project_wrong_model_deployed.md` — the central diagnosis from 2026-05-12.
- `project_model_artifact_table.md` — full hash/architecture/quant table for both models.
- `project_wrapper_status.md` — what is verified correct in the wrapper.
- `feedback_hash_before_debug.md` — the new mandatory first step.
- `feedback_verify_not_confirm.md` — don't say "this looks right" without actually tracing it.
