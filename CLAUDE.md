# CLAUDE.md — MLBuilder Project Operating Notes

## ⏸ WHERE WE LEFT OFF (2026-05-15 — drone movement script built; NEEDS Pi BLANK-mode + flight test)

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
