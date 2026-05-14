"""
tf_live_inferenceV2_gimbal_auto.py

Production live-inference + autonomous gimbal control for the flight computer.

Pulls from three pre-existing scripts:
  - test/tf_live_inferenceV2.py             -> capture/inference/writer pipeline
  - test/tf_live_infrence_gimbal_simulation -> deadband + yaw/pitch gain control law
  - test/manual_gimbal_control.py           -> MAVLink connect, stream request,
                                               telemetry ingest, transmit thread

Threads
  1. capture_thread     : RTSP -> latest_frame  (drop-old, single slot)
  2. inference_thread   : latest_frame -> detections + gimbal setpoint update
  3. gimbal_tx_thread   : periodic MAV_CMD_DO_MOUNT_CONTROL send at send/heartbeat rate
  4. main thread        : optional video writer (--no-output to skip)

Target loss behavior: HOLD. Setpoint is not modified, and the transmit thread
keeps re-sending the last (pitch, yaw) at heartbeat_send_rate.

Pi command (recommended):
  python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
      --tpu -p --no-output --mavlink tcp:10.42.0.1:5760

Bench (no autopilot present):
  python3 -B tf_live_inferenceV2_gimbal_auto.py ~/FullDataSetProd_edgetpu.tflite \
      --tpu -p --no-output --no-mavlink
"""

import sys
import time
from pathlib import Path
import threading
import cv2
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 10
WIDTH = 1920
HEIGHT = 1080

GIMBAL_YAW_MIN_DEG = -90.0
GIMBAL_YAW_MAX_DEG = 90.0
GIMBAL_PITCH_MIN_DEG = -45.0
GIMBAL_PITCH_MAX_DEG = 45.0
MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_MOUNT_MODE_MAVLINK_TARGETING = 2

COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

pipeline3 = (
    "rtspsrc location=rtsp://10.42.0.1:8554/gun_high latency=200 ! "
    "rtpjitterbuffer latency=200 ! "
    "rtph264depay ! "
    "h264parse ! "
    "avdec_h264 ! "
    "videoconvert ! "
    "appsink drop=true max-buffers=1 sync=false"
)

pipeline4 = (
    "appsrc is-live=true do-timestamp=true block=false max-bytes=20000000 format=time ! "
    "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
    "videoconvert ! "
    "video/x-raw,format=I420 ! "
    "rtpvrawpay ! "
    "udpsink host=10.42.0.1 port=7001 sync=false async=false"
)


def load_labels(label_path) -> list:
    if label_path is None:
        return COCO80_NAMES
    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [line for line in labels if line]
    if not labels:
        print(f"Warning: Labels file '{path}' is empty. Falling back to COCO names.")
        return COCO80_NAMES
    return labels


def resolve_label(class_id: int, labels: list) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


def apply_clahe_bgr(frame, clip_limit=2.0, grid_size=8):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def filter_detections(detections, frame, min_conf=0.05, max_area_ratio=0.35, edge_margin_ratio=0.01):
    h, w = frame.shape[:2]
    frame_area = float(w * h)
    mx = int(w * edge_margin_ratio)
    my = int(h * edge_margin_ratio)

    kept = []
    for d in detections:
        conf = float(d.get("confidence", 0.0))
        if conf < min_conf:
            continue

        (x1, y1), (x2, y2) = d["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        aspect = bw / max(1.0, bh)
        if aspect < 0.65 or aspect > 1.45:
            continue

        area_ratio = (bw * bh) / frame_area
        touches_edge = (x1 <= mx) or (y1 <= my) or (x2 >= (w - 1 - mx)) or (y2 >= (h - 1 - my))

        if area_ratio > max_area_ratio:
            continue
        if touches_edge and area_ratio > 0.15:
            continue

        kept.append(d)

    return kept


def crop_at(frame, ratio: float, center_x: float, center_y: float):
    h, w = frame.shape[:2]
    cw = max(1, int(w * ratio))
    ch = max(1, int(h * ratio))

    cx = int(round(center_x * (w - 1)))
    cy = int(round(center_y * (h - 1)))

    x0 = cx - cw // 2
    y0 = cy - ch // 2

    x0 = max(0, min(x0, w - cw))
    y0 = max(0, min(y0, h - ch))

    return frame[y0:y0 + ch, x0:x0 + cw], x0, y0


def remap_detections(detections, x_off: int, y_off: int):
    out = []
    for d in detections:
        (x1, y1), (x2, y2) = d["bbox"]
        c = dict(d)
        c["bbox"] = ((int(x1) + x_off, int(y1) + y_off), (int(x2) + x_off, int(y2) + y_off))
        out.append(c)
    return out


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def direction_label(err_x: float, err_y: float, deadband: float) -> str:
    horizontal = "CENTERED"
    vertical = "CENTERED"

    if err_x > deadband:
        horizontal = "RIGHT"
    elif err_x < -deadband:
        horizontal = "LEFT"

    if err_y > deadband:
        vertical = "DOWN"
    elif err_y < -deadband:
        vertical = "UP"

    if horizontal == "CENTERED" and vertical == "CENTERED":
        return "CENTERED"
    if horizontal == "CENTERED":
        return vertical
    if vertical == "CENTERED":
        return horizontal
    return f"{horizontal}+{vertical}"


def send_mount_control(master, pitch_deg: float, yaw_deg: float) -> None:
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_DO_MOUNT_CONTROL,
        0,
        float(pitch_deg),
        0.0,
        float(yaw_deg),
        0.0,
        0.0,
        0.0,
        float(MAV_MOUNT_MODE_MAVLINK_TARGETING),
    )


def request_streams(master, rate_hz: int) -> None:
    from pymavlink import mavutil

    try:
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            int(rate_hz),
            1,
        )
    except Exception as e:
        print(f"[MAVLINK] request_data_stream failed: {e}", flush=True)


class GimbalLink:
    """Owns the MAVLink connection, the current setpoint, and the transmit cadence.

    Threading model:
      - inference_thread calls update_setpoint() whenever a new detection updates yaw/pitch
      - gimbal_tx_thread (driven by run_tx_loop) wakes up at send_interval, transmits if
        dirty, otherwise re-sends the last setpoint at heartbeat_send_interval to keep
        the gimbal driver alive (same pattern as manual_gimbal_control.py).
    """

    def __init__(self, master, send_rate_hz: float, heartbeat_send_rate_hz: float, dry_run: bool):
        self.master = master
        self.dry_run = dry_run
        self.send_interval = 1.0 / float(send_rate_hz)
        self.heartbeat_send_interval = 1.0 / float(heartbeat_send_rate_hz)

        self._lock = threading.Lock()
        self._pitch = 0.0
        self._yaw = 0.0
        self._dirty = True

        self.send_count = 0
        self.last_send_time = None
        self.last_send_err = None

    def update_setpoint(self, pitch: float, yaw: float) -> None:
        with self._lock:
            if pitch != self._pitch or yaw != self._yaw:
                self._pitch = pitch
                self._yaw = yaw
                self._dirty = True

    def get_setpoint(self) -> tuple:
        with self._lock:
            return self._pitch, self._yaw

    def _transmit(self, pitch: float, yaw: float) -> None:
        if self.dry_run or self.master is None:
            self.last_send_time = time.monotonic()
            self.send_count += 1
            return
        try:
            send_mount_control(self.master, pitch, yaw)
            self.last_send_err = None
        except Exception as e:
            self.last_send_err = str(e)
            print(f"[MAVLINK] send_mount_control failed: {e}", flush=True)
        self.last_send_time = time.monotonic()
        self.send_count += 1

    def run_tx_loop(self, stop_event: threading.Event) -> None:
        last_send = 0.0
        last_heartbeat_send = 0.0

        self._transmit(self._pitch, self._yaw)
        last_send = time.monotonic()
        last_heartbeat_send = last_send

        while not stop_event.is_set():
            now = time.monotonic()
            with self._lock:
                dirty = self._dirty
                pitch = self._pitch
                yaw = self._yaw
                if dirty:
                    self._dirty = False

            if dirty and (now - last_send) >= self.send_interval:
                self._transmit(pitch, yaw)
                last_send = now
                last_heartbeat_send = now
            elif (now - last_heartbeat_send) >= self.heartbeat_send_interval:
                self._transmit(pitch, yaw)
                last_send = now
                last_heartbeat_send = now

            time.sleep(0.005)

    def run_rx_loop(self, stop_event: threading.Event) -> None:
        """Drain MAVLink inbox so the socket buffer doesn't fill up. Light logging
        for COMMAND_ACK and STATUSTEXT only; everything else is silently consumed.
        """
        if self.master is None:
            return
        while not stop_event.is_set():
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.5)
            except Exception as e:
                print(f"[MAVLINK] recv_match failed: {e}", flush=True)
                time.sleep(0.2)
                continue
            if msg is None or msg.get_type() == "BAD_DATA":
                continue
            t = msg.get_type()
            if t == "COMMAND_ACK":
                try:
                    if int(msg.command) == MAV_CMD_DO_MOUNT_CONTROL:
                        print(f"[ACK] cmd={msg.command} result={msg.result}", flush=True)
                except Exception:
                    pass
            elif t == "STATUSTEXT":
                text = (getattr(msg, "text", "") or "").strip()
                if text:
                    print(f"[STATUSTEXT sev={getattr(msg, 'severity', '?')}] {text}", flush=True)


def center_gimbal(master, duration: float, rate_hz: float, dry_run: bool) -> None:
    """Send (pitch=0, yaw=0) at rate_hz for duration seconds, then return.

    Repeats are needed because the autopilot/gimbal driver expects continuous
    MAV_CMD_DO_MOUNT_CONTROL traffic to hold the setpoint; a single packet may
    be dropped or arrive before the gimbal is armed.
    """
    interval = 1.0 / max(0.1, float(rate_hz))
    sends = max(1, int(round(duration / interval)))
    print(f"[CENTER] target (pitch=0.00, yaw=0.00) sends={sends} rate={rate_hz}Hz "
          f"duration={duration:.2f}s dry_run={dry_run}", flush=True)
    for i in range(sends):
        if dry_run or master is None:
            print(f"[CENTER] DRY {i+1}/{sends} CMD_LONG cmd={MAV_CMD_DO_MOUNT_CONTROL} "
                  f"param1=0.00 param3=0.00 param7={MAV_MOUNT_MODE_MAVLINK_TARGETING}",
                  flush=True)
        else:
            try:
                send_mount_control(master, 0.0, 0.0)
                print(f"[CENTER] TX  {i+1}/{sends} pitch=0.00 yaw=0.00", flush=True)
            except Exception as e:
                print(f"[CENTER] send_mount_control failed: {e}", flush=True)
        time.sleep(interval)
    print("[CENTER] done", flush=True)


def open_mavlink(connection_str: str, heartbeat_timeout: float, stream_rate: int):
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[MAVLINK] pymavlink not installed. Re-run with --no-mavlink.", flush=True)
        return None

    print(f"[MAVLINK] Connecting: {connection_str}", flush=True)
    try:
        master = mavutil.mavlink_connection(connection_str)
        master.wait_heartbeat(timeout=heartbeat_timeout)
        print(
            f"[MAVLINK] heartbeat sysid={master.target_system} "
            f"compid={master.target_component}",
            flush=True,
        )
        request_streams(master, stream_rate)
        return master
    except Exception as e:
        print(f"[MAVLINK] WARNING: connection/heartbeat failed: {e}", flush=True)
        print("[MAVLINK] Continuing without gimbal transmission.", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(prog="tflive_gimbal_auto",
                                     description="TF Live Inference + autonomous gimbal control")
    parser.add_argument("model", nargs="?", help="TFlite model to run", type=str)
    parser.add_argument("--nms", "-n", action="store_false")
    parser.add_argument("--tpu", "-t", action="store_true")
    parser.add_argument("--confidence", "-c", type=float, default=0.01)
    parser.add_argument("--labels", "-l", type=str, default=None)
    parser.add_argument("--process", "-p", action="store_true", help="Enable inference processing")
    parser.add_argument("--overlay", "-o", action="store_true", help="Draw detection labels/boxes on output video")
    parser.add_argument("--no-output", action="store_true", help="Disable video output stream (capture + inference only, no writer)")

    parser.add_argument("--min-conf", type=float, default=0.05, help="Post-filter minimum confidence")
    parser.add_argument("--max-area-ratio", type=float, default=0.35, help="Reject boxes larger than this frame-area ratio")
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01, help="Edge margin ratio for edge-touch rejection")

    parser.add_argument("--center-crop-pass", action="store_true", help="Run second inference pass on a crop")
    parser.add_argument("--center-crop-ratio", type=float, default=0.5, help="Primary crop ratio (0<ratio<=1)")
    parser.add_argument("--crop-center-x", type=float, default=0.5, help="Primary crop center x in [0,1]")
    parser.add_argument("--crop-center-y", type=float, default=0.5, help="Primary crop center y in [0,1]")

    parser.add_argument("--second-crop-pass", action="store_true", help="Run third inference pass on a second crop")
    parser.add_argument("--second-crop-ratio", type=float, default=0.45, help="Second crop ratio (0<ratio<=1)")
    parser.add_argument("--second-crop-center-x", type=float, default=0.55, help="Second crop center x in [0,1]")
    parser.add_argument("--second-crop-center-y", type=float, default=0.35, help="Second crop center y in [0,1]")

    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement before inference")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE grid size")

    parser.add_argument("--mavlink", type=str, default="tcp:10.42.0.1:5760",
                        help="pymavlink connection string (default: tcp:10.42.0.1:5760)")
    parser.add_argument("--no-mavlink", action="store_true",
                        help="Skip MAVLink connect; print-only dry-run (bench).")
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0,
                        help="Seconds to wait for first heartbeat (default: 15).")
    parser.add_argument("--stream-rate", type=int, default=10,
                        help="Requested telemetry stream rate from autopilot, Hz (default: 10).")
    parser.add_argument("--send-rate", type=float, default=20.0,
                        help="Max gimbal command send rate when dirty, Hz (default: 20).")
    parser.add_argument("--heartbeat-send-rate", type=float, default=2.0,
                        help="Background resend rate for the current setpoint, Hz (default: 2).")
    parser.add_argument("--center-gimbal", action="store_true",
                        help="Skip inference; just send (pitch=0, yaw=0) for --center-duration seconds and exit.")
    parser.add_argument("--center-duration", type=float, default=2.0,
                        help="Duration in seconds to repeat the center command (default: 2.0).")
    parser.add_argument("--center-rate", type=float, default=5.0,
                        help="Send rate during --center-gimbal, Hz (default: 5).")

    parser.add_argument("--deadband", type=float, default=0.08,
                        help="Normalized image-center deadband before updating gimbal angles")
    parser.add_argument("--yaw-gain", type=float, default=12.0,
                        help="Yaw gain (deg per frame at full-scale horizontal error)")
    parser.add_argument("--pitch-gain", type=float, default=10.0,
                        help="Pitch gain (deg per frame at full-scale vertical error)")
    parser.add_argument("--initial-pitch", type=float, default=0.0)
    parser.add_argument("--initial-yaw", type=float, default=0.0)

    args = parser.parse_args()

    if args.process and not args.model:
        parser.error("--process requires a model argument")

    if not (0.0 < args.center_crop_ratio <= 1.0):
        parser.error("--center-crop-ratio must be > 0 and <= 1")
    if not (0.0 < args.second_crop_ratio <= 1.0):
        parser.error("--second-crop-ratio must be > 0 and <= 1")

    for n, v in (
        ("--crop-center-x", args.crop_center_x),
        ("--crop-center-y", args.crop_center_y),
        ("--second-crop-center-x", args.second_crop_center_x),
        ("--second-crop-center-y", args.second_crop_center_y),
    ):
        if not (0.0 <= v <= 1.0):
            parser.error(f"{n} must be in [0,1]")

    if args.deadband < 0.0 or args.deadband >= 1.0:
        parser.error("--deadband must be >= 0.0 and < 1.0")
    if args.send_rate <= 0 or args.heartbeat_send_rate <= 0:
        parser.error("--send-rate and --heartbeat-send-rate must be > 0")
    if args.center_duration <= 0 or args.center_rate <= 0:
        parser.error("--center-duration and --center-rate must be > 0")

    if args.center_gimbal:
        master = None
        if not args.no_mavlink:
            master = open_mavlink(args.mavlink, args.heartbeat_timeout, args.stream_rate)
        else:
            print("[MAVLINK] Disabled via --no-mavlink (centering will be dry-run only)", flush=True)
        try:
            center_gimbal(master, args.center_duration, args.center_rate,
                          dry_run=args.no_mavlink or master is None)
        finally:
            if master is not None:
                try:
                    master.close()
                except Exception:
                    pass
        return

    cap = cv2.VideoCapture(pipeline3, cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    writer = None
    if not args.no_output:
        writer = cv2.VideoWriter(pipeline4, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
        if not writer.isOpened():
            raise RuntimeError("Failed to open GStreamer VideoWriter")
    else:
        print("[NO-OUTPUT] writer disabled; capture + inference only", flush=True)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        sys.exit(1)

    if not args.process:
        print("[ERROR] This script requires --process for gimbal automation. "
              "Use tf_live_inferenceV2.py for raw passthrough.", flush=True)
        sys.exit(2)

    try:
        labels = load_labels(args.labels)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    from MLBuilder.model.tflite.tflitemodel import TFLiteModel
    m = TFLiteModel(args.model)
    m.allocate(tpu=args.tpu)

    master = None
    if not args.no_mavlink:
        master = open_mavlink(args.mavlink, args.heartbeat_timeout, args.stream_rate)
    else:
        print("[MAVLINK] Disabled via --no-mavlink (no commands will be sent)", flush=True)

    print("[INFO] Axis assumptions:", flush=True)
    print("  image: +x=right, +y=down, center=(frame_w/2, frame_h/2)", flush=True)
    print("  gimbal yaw (param3): +right/clockwise, -left/counter-clockwise", flush=True)
    print("  gimbal pitch (param1): +up, -down", flush=True)
    print("  control: target right -> yaw+, target left -> yaw-, "
          "target down -> pitch-, target up -> pitch+", flush=True)

    link = GimbalLink(
        master=master,
        send_rate_hz=args.send_rate,
        heartbeat_send_rate_hz=args.heartbeat_send_rate,
        dry_run=args.no_mavlink or master is None,
    )

    current_pitch = clamp(args.initial_pitch, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
    current_yaw = clamp(args.initial_yaw, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
    link.update_setpoint(current_pitch, current_yaw)

    latest_frame = [None]
    latest_detections = [[]]
    frame_lock = threading.Lock()
    det_lock = threading.Lock()
    stop_event = threading.Event()
    frame_event = threading.Event()
    frame_seq = [0]

    gimbal_state = {"pitch": current_pitch, "yaw": current_yaw, "frame_idx": 0}
    gimbal_state_lock = threading.Lock()

    def capture_thread():
        skip = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                skip += 1
                if skip >= 5:
                    stop_event.set()
                time.sleep(0.01)
                continue
            skip = 0
            with frame_lock:
                latest_frame[0] = frame
                frame_seq[0] += 1
            frame_event.set()

    def inference_thread():
        nonlocal current_pitch, current_yaw
        count = 0
        last_seen = 0
        gimbal_frame_idx = 0
        while not stop_event.is_set():
            frame_event.wait(timeout=0.1)
            frame_event.clear()

            with frame_lock:
                frame = latest_frame[0]
                seq = frame_seq[0]
            if frame is None or seq == last_seen:
                continue
            last_seen = seq

            infer_frame = frame
            if args.clahe:
                infer_frame = apply_clahe_bgr(
                    infer_frame, clip_limit=args.clahe_clip_limit,
                    grid_size=args.clahe_grid,
                )

            raw = m.detect(infer_frame, nms=args.nms, tol=args.confidence)

            if args.center_crop_pass:
                crop, x_off, y_off = crop_at(
                    infer_frame, args.center_crop_ratio,
                    args.crop_center_x, args.crop_center_y,
                )
                raw_crop = m.detect(crop, nms=args.nms, tol=args.confidence)
                raw.extend(remap_detections(raw_crop, x_off, y_off))

            if args.second_crop_pass:
                crop2, x_off2, y_off2 = crop_at(
                    infer_frame, args.second_crop_ratio,
                    args.second_crop_center_x, args.second_crop_center_y,
                )
                raw_crop2 = m.detect(crop2, nms=args.nms, tol=args.confidence)
                raw.extend(remap_detections(raw_crop2, x_off2, y_off2))

            detections = filter_detections(
                raw, frame,
                min_conf=args.min_conf,
                max_area_ratio=args.max_area_ratio,
                edge_margin_ratio=args.edge_margin_ratio,
            )

            count += 1
            if count % 30 == 0:
                print(f"infer_frames={count} dets={len(detections)}", flush=True)

            with det_lock:
                latest_detections[0] = detections

            gimbal_frame_idx += 1
            if detections:
                selected = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
                (x1, y1), (x2, y2) = selected["bbox"]
                frame_h, frame_w = frame.shape[:2]
                fx = frame_w / 2.0
                fy = frame_h / 2.0
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                err_x = (cx - fx) / max(1.0, fx)
                err_y = (cy - fy) / max(1.0, fy)
                move_label = direction_label(err_x, err_y, args.deadband)

                if move_label != "CENTERED":
                    current_yaw += args.yaw_gain * err_x
                    current_pitch -= args.pitch_gain * err_y
                    current_yaw = clamp(current_yaw, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
                    current_pitch = clamp(current_pitch, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
                    link.update_setpoint(current_pitch, current_yaw)

                with gimbal_state_lock:
                    gimbal_state["pitch"] = current_pitch
                    gimbal_state["yaw"] = current_yaw
                    gimbal_state["frame_idx"] = gimbal_frame_idx

                confidence = float(selected.get("confidence", 0.0))
                print(
                    f"[F{gimbal_frame_idx:06d}] target_bbox=(({int(x1)},{int(y1)}),({int(x2)},{int(y2)})) "
                    f"center=({int(round(cx))},{int(round(cy))}) "
                    f"err=({err_x:+.3f},{err_y:+.3f}) conf={confidence:.3f} dir={move_label} "
                    f"yaw={current_yaw:.2f} pitch={current_pitch:.2f} "
                    f"CMD_LONG cmd={MAV_CMD_DO_MOUNT_CONTROL} "
                    f"param1={current_pitch:.2f} param2=0.00 param3={current_yaw:.2f} "
                    f"param4=0.00 param5=0.00 param6=0.00 "
                    f"param7={MAV_MOUNT_MODE_MAVLINK_TARGETING}",
                    flush=True,
                )
            else:
                print(f"[F{gimbal_frame_idx:06d}] NO_TARGET hold "
                      f"yaw={current_yaw:.2f} pitch={current_pitch:.2f}", flush=True)

    t_cap = threading.Thread(target=capture_thread, daemon=True)
    t_inf = threading.Thread(target=inference_thread, daemon=True)
    t_tx = threading.Thread(target=link.run_tx_loop, args=(stop_event,), daemon=True)
    t_rx = threading.Thread(target=link.run_rx_loop, args=(stop_event,), daemon=True)
    t_cap.start()
    t_inf.start()
    t_tx.start()
    if master is not None:
        t_rx.start()

    frame_interval = 1.0 / FPS

    try:
        if args.no_output:
            while not stop_event.is_set():
                time.sleep(0.5)
        else:
            while not stop_event.is_set():
                loop_start = time.monotonic()

                with frame_lock:
                    frame = latest_frame[0]
                if frame is None:
                    time.sleep(0.005)
                    continue
                frame = frame.copy()

                with det_lock:
                    detections = latest_detections[0]

                selected = None
                if detections:
                    selected = max(detections, key=lambda d: float(d.get("confidence", 0.0)))

                if args.overlay:
                    for detection in detections:
                        bbox = detection["bbox"]
                        x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
                        x_max, y_max = int(bbox[1][0]), int(bbox[1][1])
                        class_id = int(detection.get("id", -1))
                        confidence = float(detection.get("confidence", 0.0))
                        label = resolve_label(class_id, labels)

                        color = (0, 255, 0) if detection is not selected else (0, 0, 255)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        text = f"{label} {confidence:.2f}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x_min, y_min - th - 6), (x_min + tw, y_min), color, -1)
                        cv2.putText(frame, text, (x_min, y_min - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    frame_h, frame_w = frame.shape[:2]
                    fx = int(frame_w / 2.0)
                    fy = int(frame_h / 2.0)
                    cv2.drawMarker(frame, (fx, fy), (255, 255, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    with gimbal_state_lock:
                        gp = gimbal_state["pitch"]
                        gy = gimbal_state["yaw"]
                    if selected is not None:
                        (x1, y1), (x2, y2) = selected["bbox"]
                        tcx = int(round((x1 + x2) / 2.0))
                        tcy = int(round((y1 + y2) / 2.0))
                        cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)
                        cv2.line(frame, (fx, fy), (tcx, tcy), (0, 0, 255), 2)
                        cv2.putText(frame, f"yaw={gy:+.1f} pitch={gp:+.1f}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f"NO_TARGET (hold yaw={gy:+.1f} pitch={gp:+.1f})",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 200, 255), 2, cv2.LINE_AA)

                if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                    frame = cv2.resize(frame, (WIDTH, HEIGHT))
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = frame.copy()

                writer.write(frame)

                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        stop_event.set()
        t_cap.join(timeout=2.0)
        t_inf.join(timeout=2.0)
        t_tx.join(timeout=2.0)
        if master is not None:
            t_rx.join(timeout=2.0)
        cap.release()
        if writer is not None:
            writer.release()
        if master is not None:
            try:
                master.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
