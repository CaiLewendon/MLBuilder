# Session Handoff Runbook (Detailed)

Date: 2026-05-02
Owner context: live Pi + Coral deployment with relay to laptop viewer.

## 1) What Is Working Right Now
- TPU inference allocation on Pi is working:
  - `[ALLOCATE] TPU active: True`
  - input shape `[1,640,640,3]`
- Decode-path fix is validated:
  - model output `(1,5,8400)` now routes to raw-head parser.
- Relay/viewer is known to work when relay is up and output caps match receiver.

## 2) What Is Not Stable Yet
- Quality tuning:
  - foreground clutter may be mislabeled as `Target`.
  - far/background true target can be missed.
- Some script configurations over-filter and return no detections.

## 3) Critical Invariant (Do Not Regress)
In production `TFLiteModel.detect()`:
- Keep `[N,6]` branch gate strict:
  - `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:`

If this reverts, Pi may return zero detections again.

## 4) Commands You Need

### Pi inference baseline
```bash
python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o
```

### Laptop receiver baseline (relay must be up)
```bash
gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

## 5) Fast Triage Decision Tree

### Case A: No video on laptop
1. Confirm relay process is running.
2. Confirm relay output codec/port still match receiver command.
3. Confirm Pi is actually writing stream to relay input.

### Case B: Video appears, no boxes
1. Check Pi logs for inference heartbeat (`infer_frames=...`).
2. If no heartbeat: inference loop stalled.
3. If heartbeat present:
   - print `raw_count` and `filtered_count`.
   - if `raw_count > 0` and `filtered_count == 0`: filters too strict.
   - if `raw_count == 0`: detection stage or crop strategy too strict.

### Case C: Boxes appear but wrong object selected
1. Lower reliance on single-frame confidence.
2. Add/enable temporal confirmation.
3. Tune geometry/aspect/area filters gradually.
4. Bias crop strategy toward expected target region.

## 6) Recommended Next Experiment Plan
Run 5-minute fixed-scene tests and log counts:
1. Permissive baseline:
   - low model confidence threshold
   - minimal post-filter
2. Add filter thresholds one at a time:
   - min confidence
   - max area ratio
   - edge-touch suppression
3. Add crop pass:
   - center crop
4. Add upper crop:
   - target likely appears in upper middle background
5. Compare:
   - true target hit rate
   - false positive rate
   - time-to-first-detection

## 7) Suggested Debug Metrics to Print Each N Frames
- `raw_count`
- `filtered_count`
- `crop1_count`
- `crop2_count`
- optional median confidence of kept detections

These make tuning objective instead of visual-only.

## 8) File/Artifact Notes
- Current repo has documentation updates:
  - `CODEX.md`
  - `CONTEXT.md`
  - `KANBAN.md`
  - `HANDOFF.md`
- Production script filename in use:
  - `tf_live_inferenceV2.py` on Pi
- Repo script may differ:
  - `test/tf_live_infrence.py` exists for local experimentation.

## 9) Practical Stop Conditions
- Acceptable interim operating point:
  - consistent detection of true target in background in target scenario
  - false positives below operator tolerance
  - no stream interruptions due to relay mismatch

If unreachable with runtime tuning only, move to retraining with more far-target positives and clutter negatives.

---

## 10) 2026-05-07/08 Session Closeout (Local Live + Gimbal Sim)

### What was fixed in this session
1. Fixed core bbox parsing in `MLBuilder/model/tflite/tflitemodel.py`:
   - corrected normalized/pixel `xywh` interpretation in NMS path
   - corrected OpenCV NMS input format to `xywh`
2. Updated live test script for visibility and repeatability:
   - labels loaded from `target_detector_labels.txt`
   - stable overlay/log behavior retained
3. Added/renamed gimbal simulation script:
   - `test/tf_live_infrence_gimbal_simulation.py`
   - print-only MAVLink-style gimbal commands (`MAV_CMD_DO_MOUNT_CONTROL`)
   - axis conventions printed in console

### Best command right now (keep as baseline)
```bash
venv/bin/python test/tf_live_infrence_gimbal_simulation.py --video 0
```

### Why this is the best current baseline
- Uses center-crop pass by default.
- Uses tuned confidence default of `0.20`.
- Uses latest working float16 default model path in-script.
- Uses label file by default.
- Produces stable detections and readable gimbal command outputs without requiring a MAVLink endpoint.

### Next session first task
- Implement and validate real MAVLink transmission for gimbal control on drone/sim while preserving the current detection defaults (especially center-crop pass).

---

## 11) 2026-05-12 Session Closeout (Wrong Model Deployed — Root Cause Found)

### What was actually broken (and fixed)
The Pi production was running the OLD March 13 int8 model (`e4623d5d...`) instead of the April `project1_prod`-trained model (`153b25f3...`). The newer model exists on the Pi already — `~/target_detector_int8_edgetpu.tflite` — but the production script's default model path points at the older artifact. Code, wrapper, quantization math, and calibration set were all fine for the artifact actually loaded — but the artifact itself was the wrong one.

### What is verified correct (do not modify reflexively)
- `MLBuilder/model/tflite/tflitemodel.py`: handles both `(1,5,8400)` raw-head and `(1,300,6)` postprocessed outputs. Reads quant params from the model file dynamically.
- The branch gate invariant: `if out.ndim == 2 and 6 <= out.shape[1] <= 16 and out.shape[0] > out.shape[1]:` — still in place, still correct.
- Diagnostic prints in `allocate()`: keep `[TFLITEMODEL] Loaded from:` and `[ALLOCATE] ...` to make future deployments self-identifying.

### Canonical model artifact (per user designation, 2026-05-12)
**`export/project1_prod_int8_edgetpu_compat.tflite`** (hash `153b25f30817c02075f2d8d06b8b5ea83cb0313dd3b705bd57038e6758fa39a2`) is the canonical production model. Byte-identical to three other files in the repo:
- `target_detector_int8_edgetpu.tflite`
- `export/project1_prod_full_integer_quant_edgetpu.tflite`
- `export/pi_rebuild_edgetpu/target_detector_int8_edgetpu.tflite`

This is the model trained on the 492-image `project-1-at-2026-04-12-21-16-9fb8c3ae` dataset. NMS is baked in (output is `(1, 300, 6)`).

### Production command (after model swap on Pi)
```bash
python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu \
  -l ../target_detector_labels.txt -p -o
```

If swap was via file copy (Option B in CONTEXT.md), the original baseline command still works:
```bash
python3 -B tf_live_inferenceV2.py ~/model_full_integer_quant_edgetpu.tflite --tpu \
  -l ../target_detector_labels.txt -p -o
```

### Visual signature of "right model loaded"
- `[OUT] shape=(1, 300, 6)` — this is the NEW project1_prod model
- `[OUT] shape=(1, 5, 8400)` — this is the OLD March model (do NOT ship)

### Decision tree if confidence is still weak after swap
1. Verify shape line is `(1, 300, 6)`. If not, the swap didn't take.
2. If far targets still weak: it's a training-data problem (small/far target underrepresented in 492-image set). Capture more far-distance positives and hard-negative clutter from the deployment scene. Retrain `project1_prod.pt`, re-export.
3. If false positives on clutter persist: add clutter scenes (target-absent) to training. Same retrain path.
4. Optional intermediate: rebuild int8 with 492-image calibration instead of 99 (val-only). Artifacts ready at `project-1-at-2026-04-12-21-16-9fb8c3ae/data_calib.yaml`; requires `edgetpu_compiler` on laptop.

### Lessons captured for future debugging
- Always hash the deployed model before assuming code or quantization is the cause. The `sha256sum` step took 5 seconds and would have saved this entire investigation.
- Two different models with similar names (`*_full_integer_quant_edgetpu.tflite`) caused the confusion. Prefer the explicit name `target_detector_int8_edgetpu.tflite` going forward.
- Diagnostic prints in the wrapper (model path, quant params) are cheap and high-leverage; keep them.

---

## 12) 2026-05-13 Session Closeout — FullDataSetProd: Trained, Exported, **Compiled**, Awaiting Pi Deploy

### What was accomplished
Trained the successor production model `FullDataSetProd` from a 3,487-image Label Studio export (7× the prior 492). Final val P=0.956, R=0.85, mAP50=0.951, mAP50-95=0.719. All int8 + float TFLite variants exported. **EdgeTPU compile finished 2026-05-13 17:24 UTC** after working around a TF 2.19 / edgetpu_compiler 16.0 op-version incompatibility (see § 12a). Pi deploy + verification is the only remaining step.

### Canonical EdgeTPU artifact (compiled this session)
- `export/FullDataSetProd_full_integer_quant_edgetpu.tflite` (3.05 MiB)
- sha256 `3d599378240246a660d707839aee6506bfaa44b1e135a354fc13b04dfda0d3f3`
- Input/output: int8, scale 1/255, zero -128
- Output: `(1, 5, 8400)` raw head (intentional — `nms=False`)
- Pre-compile compatible file: `export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_dwfix_v3.tflite` sha256 `b62b0f55ef94619d26f9ec2ff1db2cffb54617301dea388c2ef19f593c9b7f5d`

### Resume-from-here checklist (steps 1 & 2 already done this session, see § 12a for details)
1. ~~Install `edgetpu_compiler` on laptop~~ — **DONE** 2026-05-13. Installed via direct `.deb` (apt repo doesn't ship `noble` codename). Modern keyring path:
   ```bash
   curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu.gpg
   curl -fsSL -o /tmp/edgetpu-compiler.deb https://packages.cloud.google.com/apt/pool/coral-edgetpu-stable/edgetpu-compiler_16.0_amd64_3ccd3b6ea6298eaaae6aa045764b3184.deb
   sudo apt install -y /tmp/edgetpu-compiler.deb
   ```
   Verify: `edgetpu_compiler --version` → `Edge TPU Compiler version 16.0.384591198`.
2. ~~Compile~~ — **DONE** 2026-05-13. NOT a simple one-shot: required TF 2.15 sidecar venv + flatbuffer surgery to convert one grouped-CONV_2D node back to DEPTHWISE_CONV_2D. Full recipe in § 12a. Output: `export/FullDataSetProd_full_integer_quant_edgetpu.tflite` sha256 `3d599378240246a660d707839aee6506bfaa44b1e135a354fc13b04dfda0d3f3`.
3. **Back up current Pi model + scp new one** (new file name to avoid the "two models with same name" confusion that burned the 2026-05-12 session):
   ```bash
   ssh pi@<PI_IP> 'cp ~/target_detector_int8_edgetpu.tflite ~/target_detector_int8_edgetpu.tflite.PROJECT1PROD_153b25f3_BACKUP'
   scp export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_edgetpu.tflite \
       pi@<PI_IP>:~/FullDataSetProd_edgetpu.tflite
   ```
4. **Run on Pi** (new file path — do NOT just overwrite the prior file):
   ```bash
   python3 -B tf_live_inferenceV2.py ~/FullDataSetProd_edgetpu.tflite --tpu \
     -l ../target_detector_labels.txt -p -o
   ```
5. **Verify** in logs:
   - `[TFLITEMODEL] Loaded from: /home/pi/FullDataSetProd_edgetpu.tflite`
   - `[OUT] shape=(1, 5, 8400)` ← **NEW signature, raw-head; do NOT panic, this is correct for FullDataSetProd**
   - `[ALLOCATE] TPU active: True`
   - Hash the deployed file: `sha256sum ~/FullDataSetProd_edgetpu.tflite` — record it for future deployment-identity checks. Source hash (pre-compile): `db579abd1359952f48b9cb4e83c3ffaf3836c080e683bb04c25b8fa6b87f1534`. The edgetpu_compiler produces a slightly different file, so the deployed hash will differ — but should be byte-identical to whatever the compile produces locally; capture both.

### Decision still open
- **Output-shape signature**: FullDataSetProd is `(1, 5, 8400)` raw head (due to `nms=False`). Prior canonical was `(1, 300, 6)` postprocessed. If you want the visual fingerprint back, re-export with `nms=True`:
  ```bash
  venv/bin/yolo export model=export/FullDataSetProd.pt format=tflite int8=True imgsz=640 \
    data=project-1-at-2026-05-13-06-40-8e81e090/data_calib_subset.yaml nms=True
  ```
  Then re-compile. Costs ~35 min of int8 calibration. Alternative: accept raw-head and use sha256 as deployment identity (recommended — hash is more robust than shape).

### If FullDataSetProd performs worse than project1_prod in the field
- The project1_prod (`153b25f3`) model is still on the Pi at `~/target_detector_int8_edgetpu.tflite` until you overwrite it. Rollback = one scp away (or `ssh pi 'cp ~/target_detector_int8_edgetpu.tflite.PROJECT1PROD_153b25f3_BACKUP ~/target_detector_int8_edgetpu.tflite'` after step 3).
- The new dataset has a 990-image near-duplicate cluster (kept in train), so val (697 images) is a pessimistic benchmark. Real-world performance may track val OR be moderately better.

### Repro recipes saved (for future retrains on new datasets)
- Scene-aware split: `venv/bin/python test/prepare_dataset_split.py --dataset <NEW_DATASET_DIR>` — writes `data.yaml`, `train.txt`, `val.txt`, `calib_all.txt`, `data_calib.yaml` in one shot.
- Calibration cap: always produce a 500-image subset for int8 export on 16 GB laptops: `shuf --random-source=<(yes 0) -n 500 calib_all.txt > calib_subset_500.txt`, then a `data_calib_subset.yaml` pointing at it.
- EdgeTPU op-version incompatibility workaround: see § 12a below.

## 12a) EdgeTPU Compile Workaround — TF 2.19 vs `edgetpu_compiler` 16.0 Op-Version Gap

**Problem.** Ultralytics' `yolo export ... int8=True` in this venv (TF 2.19, onnx2tf 1.28.8) emits `CONV_2D` op-code version **6** and collapses one depthwise conv into a grouped CONV_2D. `edgetpu_compiler` 16.0 (the latest Coral release, August 2022) only supports CONV_2D up to version 5 and rejects grouped CONV_2D entirely. Both errors observed this session:
- `ERROR: Didn't find op for builtin opcode 'CONV_2D' version '6'`
- After version-flag downgrade: `ERROR: :349 input->dims->data[3] != filter->dims->data[3] (128 != 1)` at node 145 — a CONV_2D with `groups=128, in_per_g=1, C_out=128`, mathematically a depthwise conv mis-encoded as grouped.

**What didn't work**
- Trying the float32-IO `_int8.tflite` or `_integer_quant.tflite` variants (same op version).
- `onnx2tf -dgc` (disable group convolution) — produced malformed convs that TF's int8 calibrator rejects with `input_channel % filter_input_channel != 0 (1 != 0)`.
- TF 2.15 sidecar venv conversion alone — still emits CONV_2D v6 with the grouped-conv encoding for the one offending node.

**What worked (2-stage)**
1. **TF 2.15 sidecar venv** — needed only because TF 2.15's TFLiteConverter emits cleaner ops with regular groups=1 CONV_2D for all but one node (the same single grouped conv). Install path (no sudo for python, only for the package):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ~/.local/bin/uv python install 3.11
   ~/.local/bin/uv venv --python 3.11 venv-tf215
   ~/.local/bin/uv pip install --python venv-tf215/bin/python "tensorflow==2.15.0" "numpy<2" pillow
   venv-tf215/bin/python build/convert_int8_tf215.py
   ```
   Output: `export/FullDataSetProd_saved_model/FullDataSetProd_full_integer_quant_tf215.tflite`. Calibration uses `build/calib_500x3x640x640_float32.npy` (built once via `build/build_calib_npy.py` from `calib_subset_500.txt`).

2. **Flatbuffer surgery** to (a) convert the lone grouped CONV_2D op (filter `(128,3,3,1)`, in_c=128) back into DEPTHWISE_CONV_2D, and (b) downgrade the CONV_2D op-code version flag 6→3:
   ```bash
   venv-tf215/bin/python build/surgery_grouped_to_depthwise.py
   # produces _dwfix.tflite (still has CONV_2D version=6 flag)
   # Then a tiny inline script flips CONV_2D op-code version 6 -> 3 -> _dwfix_v3.tflite
   ```
   The math: a CONV_2D with filter `(C_out, kH, kW, 1)` and input_c == C_out is exactly a depthwise conv with depth_multiplier=1. Filter weights transpose `(C_out, kH, kW, 1) → (1, kH, kW, C_out)`, per-channel quant axis 0 → 3, op-code 3 → 4, builtinOptions `Conv2DOptionsT` → `DepthwiseConv2DOptionsT`. Verified: `tf.lite.Interpreter` runs end-to-end and produces correct `(1, 5, 8400)` output. `edgetpu_compiler` then compiles it.

**Result**: 132 ops on EdgeTPU / 211 on CPU — same shape of split as project1_prod (132/339). Final hash `3d599378240246a660d707839aee6506bfaa44b1e135a354fc13b04dfda0d3f3` at `export/FullDataSetProd_full_integer_quant_edgetpu.tflite`.

**For future retrains on this stack**: every `yolo export` here will hit the same wall. The two helper scripts (`build/convert_int8_tf215.py`, `build/surgery_grouped_to_depthwise.py`) and the sidecar venv are reusable as-is — just point them at the new saved_model / int8 TFLite. The 500-image calibration npy can be rebuilt for a new dataset via `build/build_calib_npy.py` after `prepare_dataset_split.py`.

### Files generated this session (don't lose track)
- `test/prepare_dataset_split.py` (NEW helper, reusable for future datasets)
- `project-1-at-2026-05-13-06-40-8e81e090/data.yaml` `train.txt` `val.txt` `calib_all.txt` `calib_subset_500.txt` `data_calib.yaml` `data_calib_subset.yaml`
- `export/FullDataSetProd.pt`, `FullDataSetProd_last.pt`, `FullDataSetProd.onnx`
- `export/FullDataSetProd_saved_model/` containing 5 TFLite variants + saved_model.pb
- `build/out/FullDataSetProd_train.log`, `build/out/FullDataSetProd_export_int8.log`
- All hashes in CONTEXT.md "2026-05-13 Session Addendum" table.
- Original training output (different dir): `/home/caile/Documents/MLBuilder/runs/detect/build/out/FullDataSetProd/` — keep, has `args.yaml`, `results.csv`, plots.

## 13) 2026-05-13 Evening Session — Pi Live Inference Tuning + Output Bottleneck Diagnosis

### What was accomplished
Deployed `FullDataSetProd_edgetpu.tflite` to the Pi, ran live, and went through a long debugging chain on why the output stream was bursty/freezing. Final answer: the **output stream** (raw 1080p over UDP via `rtpvrawpay`) is what's choking the Pi, not inference, not the camera, not the decoder. With `--no-output`, inference runs steady at ~7 fps with no gaps.

### Key findings (in order of discovery)
1. **Stale-frame re-processing bug**: `frame_event.wait(timeout=0.1)` returns regardless of event state. Without checking the return value, inference reprocessed the same frame multiple times in tight loops. Fixed via `frame_seq` counter + `if seq == last_seen: continue`. This is now standard in the inference thread of `tf_live_inferenceV2.py`.
2. **Inference is bounded at ~140 ms/cycle = 7 fps** on this Pi+TPU combo. Per compile log, FullDataSetProd has 132 ops on TPU and 211 on CPU (~28% TPU acceleration). The CPU postprocess + Python pre/post around `m.detect()` is the cost. No code change makes this faster short of re-exporting at `imgsz=320`.
3. **Source camera delivers 30 fps clean** — confirmed with a standalone gst+python test (`test.py` running `appsink` with `emit-signals=true` counting buffers): `26.90 fps average` over 10 s without script load.
4. **Pi 5 / Trixie OS does NOT expose `v4l2h264dec`**: `/dev/video10-12` codec nodes are not present (only `/dev/video19` rpivid metadata). The hardware H.264 path is gone; software `avdec_h264` is the only option. It keeps up with 30 fps standalone but competes for CPU when output is running.
5. **The actual choke is `videoconvert + rtpvrawpay + udpsink` in pipeline4** at 1080p/30. Raw YUV at that rate is ~250 Mbit/s and the GStreamer scheduler thread for it saturates a core. Capture's RTSP input thread gets starved → effective capture rate drops to ~1 fps → inference looks like it freezes for 7+ seconds at a time.
6. **Diagnostic that nailed it**: `--no-output` flag (added to `test/tf_live_inferenceV2.py`). With output disabled, inference fires steady at ~140 ms cadence, no multi-second gaps. With output enabled at 1080p10 (or 30), inference falls into the bursty pattern. Confirms: writer is the cause.

### Things ruled out during the debugging chain (don't re-chase)
- Network bandwidth on the wifi link — verified via raw-mode test at FPS=10 1080p, smooth.
- TPU thermal throttling — `vcgencmd get_throttled` was clean.
- GIL contention from inference Python pre/post — no thread exceeded 50% CPU in `htop`.
- USB contention between Coral and wifi — Pi has built-in wifi (PCIe), not USB-shared with Coral.
- H.264 B-frame buffering at decode — confirmed not the cause once the source standalone showed 30 fps.
- "Dual-clock waves" between writer-paced sleep loop and capture-paced frame_event — was an early diagnosis I gave but proved wrong; the actual cause was writer CPU saturation.

### The fix path (pick one)
- **Option A (recommended for now)**: Run `tf_live_inferenceV2.py` with `--no-output`. Inference at full 7 fps, detections go to MAVLink/telemetry. No video preview.
- **Option B**: H.264-encode pipeline4 to drop TX bitrate ~40× and unblock the GStreamer scheduler:
  ```python
  pipeline4 = (
      "appsrc is-live=true do-timestamp=true block=false max-bytes=20000000 format=time ! "
      "queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
      "videoconvert ! "
      "video/x-raw,format=I420 ! "
      "x264enc tune=zerolatency speed-preset=ultrafast bitrate=6000 key-int-max=30 ! "
      "rtph264pay config-interval=1 pt=96 ! "
      "udpsink host=10.42.0.1 port=7001 sync=false async=false"
  )
  ```
  Laptop side: replace `rtpvrawdepay` with `rtph264depay ! avdec_h264`.
- **Option C**: Two-machine split — Pi runs inference + sends detection-only telemetry; laptop pulls RTSP directly from the camera for preview.

### Architecture of `tf_live_inferenceV2.py` (decoupled, as currently committed)
Three independent loops:
- **`capture_thread`**: tight loop on `cap.read()`, writes `latest_frame[0]` + bumps `frame_seq[0]` under `frame_lock`, sets `frame_event`. Paced naturally by GStreamer source rate (cap.read blocks waiting).
- **`inference_thread`**: waits on `frame_event`, dedups via `seq == last_seen`, runs `m.detect(...)`, stores in `latest_detections[0]` under `det_lock`. Does NOT touch the writer. Cadence: ~7 fps (inference-bound).
- **Main thread writer loop** (only runs when `--no-output` is NOT set): paced at `1/FPS` sleep, reads `latest_frame[0]` + `latest_detections[0]`, draws overlay if `--overlay`, writes via `writer.write()`. Detections lag by ~150 ms (one inference cycle) but the *video stream* itself is at writer cadence.

### Verifying source-side rate independently
File: `test.py` on Pi (or laptop). Standalone python-gstreamer counter:
```python
import gi, time
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
p = Gst.parse_launch('rtspsrc location=rtsp://10.42.0.1:8554/front_high latency=200 ! rtpjitterbuffer latency=200 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink name=s emit-signals=true sync=false drop=false max-buffers=1')
sink = p.get_by_name('s')
n = [0]
def on_sample(s):
    n[0] += 1
    s.emit('pull-sample')
    return 0
sink.connect('new-sample', on_sample)
p.set_state(Gst.State.PLAYING)
t0 = time.monotonic()
try:
    while time.monotonic() - t0 < 10:
        time.sleep(1)
        e = time.monotonic() - t0
        print(f'{e:4.1f}s: frames={n[0]:4d}  fps={n[0]/e:5.2f}')
finally:
    p.set_state(Gst.State.NULL)
```
Run with `/usr/bin/python3 test.py` (NOT pyenv's python — pyenv has a broken `gi` package shadowing the real PyGObject. See `which gi` to confirm; system path is `/usr/lib/python3/dist-packages/gi/`).

### Files this session
- `test/tf_live_inferenceV2.py` — updated with seq-dedup + `--no-output` flag + decoupled writer architecture.
- `test/tf_live_inferenceV2_backup.py` — pre-`--no-output` reference copy for diffing.
- On Pi: `~/FullDataSetProd_edgetpu.tflite` (deployed). Original `target_detector_int8_edgetpu.tflite` (project1_prod 153b25f3) still present as a fallback.

### Pi commands cheatsheet
```bash
# Recommended: inference only, no output stream
python3 -B tf_live_inferenceV2.py ~/FullDataSetProd_edgetpu.tflite --tpu -p --no-output

# With output (currently choky at 1080p raw — needs H.264 pipeline4 fix)
python3 -B tf_live_inferenceV2.py ~/FullDataSetProd_edgetpu.tflite --tpu -p -o

# Fallback to project1_prod
python3 -B tf_live_inferenceV2.py ~/target_detector_int8_edgetpu.tflite --tpu -l ../target_detector_labels.txt -p -o
```

## § 14 — Autonomous gimbal automation script (`tf_live_inferenceV2_gimbal_auto.py`, 2026-05-14)

### Purpose
Production live-inference + autonomous gimbal tracking in one script. Replaces the need to run `tf_live_inferenceV2.py` + `tf_live_infrence_gimbal_live.py` separately for the gimbal-tracking use case. User confirmed end-to-end working ("works amazingly").

### Pi run modes
```bash
# Live autonomous tracking (production)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
    --tpu -p --no-output --mavlink tcp:10.42.0.1:5760

# Bench dry-run (no autopilot connected)
python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
    --tpu -p --no-output --no-mavlink

# Send gimbal to center and exit (no model/camera required)
python3 -B tf_live_inferenceV2_gimbal_auto.py --center-gimbal --mavlink tcp:10.42.0.1:5760

# Local bench centering dry-run (no autopilot)
python3 -B tf_live_inferenceV2_gimbal_auto.py --center-gimbal --no-mavlink --center-duration 1.0 --center-rate 5
```

### Tuning args (next-session focus per user)
- `--deadband` (default 0.08) — increase to suppress jitter, decrease for tighter centering.
- `--yaw-gain` (default 12.0) — degrees per frame at `err_x=1.0`. Increase for snappier yaw; decrease if oscillating.
- `--pitch-gain` (default 10.0) — same for pitch.
- `--send-rate` (default 20 Hz) — max gimbal command emit rate.
- `--heartbeat-send-rate` (default 2 Hz) — periodic re-send so gimbal driver doesn't time out.

### Decision tree if it misbehaves
- **No motion at all** → check `[ACK] cmd=205 result=...` in log. result≠0 means autopilot rejected. Check `MNT1_TYPE` / `SERVOx_FUNCTION` on FC.
- **Gimbal oscillates around target** → `yaw-gain`/`pitch-gain` too high. Halve and retry.
- **Gimbal lags / drifts off** → gain too low, or `--deadband` too wide. Lower deadband first (cheaper to test).
- **Gimbal centered but log shows non-CENTERED** → `err_x` / `err_y` printed in each `[F######]` line; if they're inside ±deadband and label says otherwise, that's a deadband-comparison bug (not seen yet).
- **`[CENTER]` runs but gimbal doesn't move** → autopilot may be in a mode that ignores `MAV_CMD_DO_MOUNT_CONTROL`. Check `MNT1_DEFLT_MODE` and confirm `MAV_MOUNT_MODE_MAVLINK_TARGETING` is permitted.
- **MAVLink connect fails** → script prints warning and continues without TX (dry-run-ish). Re-check with `--mavlink udp:...` if TCP path is congested.

### What it does NOT do (known gaps)
- No FOV-calibrated gain mapping (gains are heuristic).
- No closed-loop feedback against `MOUNT_STATUS` / `GIMBAL_DEVICE_ATTITUDE_STATUS` (those are logged in `run_rx_loop` but unused).
- No earth-frame stabilization (body-frame only — airframe roll/pitch is not compensated).
- No PWM fallback (`MAV_CMD_DO_SET_SERVO` path); user briefly raised this then redirected.

### Files in scope
- `test/tf_live_inferenceV2_gimbal_auto.py` (new, ~530 lines)
- Read-only references kept for diffing: `test/tf_live_inferenceV2.py`, `test/tf_live_infrence_gimbal_simulation.py`, `test/manual_gimbal_control.py`, `test/tf_live_infrence_gimbal_live.py`
- `ailearn.sh` at repo root (new; was missing from this repo, copied from `~/Documents/aerospace2025-26/UAS_Competition_task_1_2026/CODEX/ailearn.sh`)
