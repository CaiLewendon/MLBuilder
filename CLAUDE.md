# CLAUDE.md — MLBuilder Project Operating Notes

## ⏸ WHERE WE LEFT OFF (2026-05-14 late — autonomous tracking + state-machine firing, both USER-VALIDATED)

**Two big subsystems converged this session, both confirmed working end-to-end on the Pi against the autopilot:**

### 1. Gimbal control law — slew-rate-limited proportional (final architecture)
The integrator runaway diagnosed earlier in the day was fixed by **decoupling the controller from the transmitted setpoint via `GimbalLink`**:
- Inference thread computes `wished_yaw = link.get_current() + yaw_gain * err_x` **fresh every frame** (NOT integrated). The reference is the actual transmitted position, not a free-running counter.
- TX thread (20 Hz) ramps `current_yaw` toward `wished_yaw` by at most `max_slew_rate_yaw / 20` degrees per tick. Hard rate-limit. Gimbal can never be commanded faster than its physical slew can keep up with.
- Per-frame log carries both `cur=(y,p)` (transmitted, what gimbal sees) and `wished=(y,p)` (controller intent, leads by `gain * err`).
- Defaults: `--yaw-gain 12`, `--pitch-gain 10` (sim values, unchanged), `--max-slew-rate-yaw 2.0`, `--max-slew-rate-pitch 1.5` (deg/sec).
- New `--start-from-current` flag: reads `MOUNT_STATUS` / `GIMBAL_DEVICE_ATTITUDE_STATUS` via `read_current_gimbal_position()` and starts the controller from the gimbal's actual orientation. Without the flag, the script centers the gimbal first.

### 2. Fire control — phase state machine driven by RELAY_STATUS confirmation (final architecture)
After iterating through many false starts (DO_REPEAT_RELAY with double-send, DO_SET_RELAY ON+timer+OFF, idle heartbeats, blind keepalive), we landed on:
- **Five-phase state machine**: `IDLE → ARMING → FIRING → DISARMING → COOLDOWN → IDLE`.
- **No phase advance without `RELAY_STATUS` confirmation** for ARMING→FIRING (waits for bit=1) and DISARMING→COOLDOWN (waits for bit=0). FIRING→DISARMING and COOLDOWN→IDLE are time-based.
- **2 Hz keepalive** in every phase sends `DO_SET_RELAY` with the desired state. Idempotent. Single-packet loss can't strand the relay in the wrong state for more than ~500 ms.
- **`RELAY_STATUS` (msg 376)** subscribed at 5 Hz via `MAV_CMD_SET_MESSAGE_INTERVAL` (511) at MAVLink connect. ArduPilot's relay bits are 0-indexed; gun is relay **1** (matches QGC "Shoot Gun" action's `param1=1`).
- **`fire_period` timer starts on confirmed ON** (not on send), so 5 s burst is exactly 5 s ON from autopilot's perspective. Same for `fire_cooldown` after confirmed OFF.
- BLANK mode (`--live-fire` absent) auto-confirms phase transitions after 100 ms simulated lag for testing.

### Key learnings from the long debugging arc (don't repeat)
- **`MAV_CMD_DO_REPEAT_RELAY` (182) with `cycles=1, period=N`** does **N/2 seconds ON, N/2 seconds OFF** in ArduPilot. NOT "ON for N seconds". `period=5` gave us 2.5 s ON — wrong tool for "fire for X seconds" requirements.
- **Double-sending `DO_REPEAT_RELAY` (50 ms apart)** caused on/off/on/off chatter — each command starts its own relay-pulse state machine in the autopilot, and concurrent state machines collide. **Single-send only** for that command.
- **Sending `DO_SET_RELAY` OFF while a `DO_REPEAT_RELAY` cycle is in progress** interrupts the cycle. Mixing these commands is brittle. Pick one mechanism and stick with it.
- The right primitive for "script controls the on-time" is **`DO_SET_RELAY` (181) ON, wait, `DO_SET_RELAY` OFF**. The autopilot just holds state; the script owns timing.
- **QGroundControl's "Shoot Gun" action** is `cmd=182 param1=1 param2=1 param3=2` — relay 1, 1 cycle of 2 s = 1 s pulse. Verified by user.
- **The user's autopilot publishes `RELAY_STATUS`** and our `SET_MESSAGE_INTERVAL` request works — confirmed at runtime.

### Working artifacts on disk
- `test/tf_live_inferenceV2_gimbal_auto.py` — autonomous tracking + state-machine firing (FINAL — user-confirmed working).
- `test/manual_gimbal_control.py` — `--live-fire` is now a one-shot fire-and-exit (parallel to `--center-gimbal`). Press `f` in panel mode for BLANK simulation; pass `--live-fire` as a CLI arg for real one-shot fire. Used as the isolation harness that confirmed the autopilot's relay control.

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
