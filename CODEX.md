# CODEX Session Log: Edge TPU + Live Inference Bring-Up

Date: 2026-05-02
Project: `MLBuilder`
Scope: Laptop local validation, Raspberry Pi + Coral Edge TPU deployment, RTSP ingest and UDP relay output.

## Objective
- Run target detection on live camera/RTSP using custom dataset artifacts.
- Deploy Edge TPU model to Pi and validate detections + relay stream to laptop.

## Artifacts Identified
- Dataset:
  - `project-1-at-2026-04-12-21-16-9fb8c3ae/data.yaml`
  - `project-1-at-2026-04-12-21-16-9fb8c3ae/classes.txt` (`Target`)
- Working CPU-side TFLite:
  - `model_saved_model/model_int8.tflite`
- Edge TPU model for deployment:
  - `export/model_full_integer_quant_edgetpu.tflite`
- Label file used in deployment:
  - `target_detector_labels.txt` (contains `Target`)

## Key Findings
1. `scp -i <model.tflite> ...` failed because `-i` is SSH key flag, not source-file flag.
2. `_edgetpu.tflite` cannot run on CPU-only interpreter (`edgetpu-custom-op` unresolved), expected behavior.
3. `export/project1_prod_int8.tflite` behaved inconsistently in this stack (no useful detections in app path).
4. Confirmed `model_saved_model/model_int8.tflite` produced detections locally.
5. Pi runtime showed TPU delegate active:
   - `[ALLOCATE] TPU active: True`
   - input `int8`, shape `[1,640,640,3]`
6. Major decode bug on Pi path:
   - Model output probe showed `out shape: (5, 8400)` (raw head format), not `[N,6]`.
   - Existing code routed this through `[N,6]` branch due to broad condition, causing zero detections.
7. Correcting branch gate fixed detections on Pi immediately.

## Root Cause
`detect()` postprocessed branch condition was too permissive:
- Bad condition:
  - `if out.ndim == 2 and out.shape[1] >= 6:`
- For `out.shape == (5,8400)`, this incorrectly matched and treated raw-head output as `[N,6]`, suppressing all detections.

## Production Fix That Worked
Change only the `[N,6]` gate condition in production `TFLiteModel.detect()`:

From:
- `if out.ndim == 2 and out.shape[1] >= 6:`

To:
- `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:`

Result after fix:
- Pi showed continuous non-zero detections (`detections=1..3`, periodic `infer_frames=... dets=1`).

## Verified Commands
- Pi inference (working after fix):
  - `python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o -c 0.01`
- Output-shape probe that proved raw-head format:
  - output shape `(1,5,8400)`, dequantized max ~`1.007`

## Streaming/Relay Notes
- Inference path and stream-view path are separate concerns.
- A temporary “no video” event was relay-side (relay not up), not inference-side.
- Once relay path was valid, detections were still confirmed in Pi logs.

## What Not To Change
- Do not broadly rewrite production decode logic unless needed.
- Keep production code minimal-change for this fix (single branch gate correction).

## Final Status
- Edge TPU inference path: WORKING.
- Detection outputs: WORKING on Pi.
- Remaining ops are deployment hygiene (relay uptime/caps validation, optional confidence tuning).
