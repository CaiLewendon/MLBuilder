# Kanban

## Done
- Identify candidate live inference scripts and model artifacts.
- Verify dataset split counts and custom label set presence.
- Identify Coral-compatible deployment artifact:
  - `export/model_full_integer_quant_edgetpu.tflite`
- Validate Pi TPU allocation (`TPU active: True`).
- Add runtime instrumentation to confirm inference loop activity.
- Probe raw model output on Pi and confirm shape `(1,5,8400)`.
- Isolate decode-branch bug for raw-head output.
- Apply minimal production-safe fix to `[N,6]` branch condition.
- Re-run on Pi and confirm non-zero detections.
- Confirm “video not showing” incident due to relay state mismatch (relay down), not inference.

## In Progress
- None.

## Next
- Lock relay/service startup ordering so relay is always up before inference test.
- Capture and store one known-good end-to-end test transcript (Pi logs + laptop viewer).
- Optionally tune confidence threshold (`0.01` -> operational value) based on false positive tolerance.
- Optional: add a one-line runtime print of selected decode path for future debugging.

## Backlog
- Add unit/regression test for `detect()` branch selection:
  - `[N,6]` path should require `N >> cols` shape.
  - `(5,8400)` must route to raw-head path.
- Add deployment checklist doc for model/label/relay/receiver caps alignment.
