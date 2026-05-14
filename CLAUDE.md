# CLAUDE.md — MLBuilder Project Operating Notes

## ⏸ WHERE WE LEFT OFF (2026-05-14 — autonomous gimbal script working)
**Last action:** Built `test/tf_live_inferenceV2_gimbal_auto.py` — a single production script combining V2's threaded inference, the simulation script's deadband+gain control law, and the manual-gimbal script's MAVLink connection/telemetry/transmit threading. User ran it end-to-end and confirmed it **works**. Quote: *"that script works amazingly. it just needs some logic tuning and gain control on the movement and mapping, but other than that its great."* Added a `--center-gimbal` early-exit mode that skips inference and just emits `MAV_CMD_DO_MOUNT_CONTROL(pitch=0, yaw=0)` for a configurable duration. Also installed `ailearn.sh` at repo root (was missing from this repo though present in sibling projects).

**What's left (next session):** tune `--yaw-gain` (default 12) and `--pitch-gain` (default 10) — they're heuristic, not FOV-calibrated. Possibly add FOV-based mapping (`pixel_err × HFOV/2 = deg_err`), gimbal feedback closed-loop via `MOUNT_STATUS`/`GIMBAL_DEVICE_ATTITUDE_STATUS`, or earth-frame stabilization. **User briefly asked about PWM-based centering then redirected to mapping discussion — PWM option still on the table.**

**Working artifacts on disk:**
- `test/tf_live_inferenceV2_gimbal_auto.py` — new production script (this session).
- `test/tf_live_inferenceV2.py` — prior V2 inference (still valid for inference-only deployments without gimbal).
- `ailearn.sh` at repo root (added this session; copy of UAS_Competition_task_1_2026 version).

**Pi commands**:
```bash
# Live autonomous gimbal tracking (production)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite --tpu -p --no-output --mavlink tcp:10.42.0.1:5760

# Bench dry-run (no autopilot)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite --tpu -p --no-output --no-mavlink

# Center gimbal and exit (no model/camera needed)
python3 -B tf_live_inferenceV2_gimbal_auto.py --center-gimbal --mavlink tcp:10.42.0.1:5760
```

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
