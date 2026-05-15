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
MAV_CMD_DO_SET_RELAY = 181
MAV_CMD_DO_REPEAT_RELAY = 182
MAV_CMD_REQUEST_MESSAGE = 512
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAVLINK_MSG_ID_RELAY_STATUS = 376

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
    "rtspsrc location=rtsp://10.42.0.1:8554/front_high latency=200 ! "
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


def send_relay(master, relay_num: int, state: int) -> None:
    """MAV_CMD_DO_SET_RELAY (181). Used here as a SAFETY OFF after a fire burst
    completes, and on script exit."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_DO_SET_RELAY,
        0,
        float(int(relay_num)),
        float(int(bool(state))),
        0.0, 0.0, 0.0, 0.0, 0.0,
    )


def send_repeat_relay(master, relay_num: int, cycles: int, cycle_time_s: float) -> None:
    """MAV_CMD_DO_REPEAT_RELAY (182). Matches the QGroundControl 'Shoot Gun'
    action: param1=relay, param2=cycles, param3=cycle_time. ArduPilot owns the
    pulse timing — the script just re-issues the command when continuous fire
    is desired (i.e. target stays centered past the burst duration)."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_DO_REPEAT_RELAY,
        0,
        float(int(relay_num)),
        float(int(cycles)),
        float(cycle_time_s),
        0.0, 0.0, 0.0, 0.0,
    )


def set_message_interval(master, msg_id: int, interval_us: int) -> None:
    """MAV_CMD_SET_MESSAGE_INTERVAL (511). Request streaming of msg_id at interval."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(int(msg_id)),
        float(int(interval_us)),
        0.0, 0.0, 0.0, 0.0, 0.0,
    )


def request_message(master, msg_id: int) -> None:
    """MAV_CMD_REQUEST_MESSAGE (512). Ask the autopilot to send msg_id once."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_REQUEST_MESSAGE,
        0,
        float(int(msg_id)),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
    """Owns the MAVLink connection. Holds two setpoints:

      - wished_pitch / wished_yaw : where the controller wants the gimbal (inference thread)
      - current_pitch / current_yaw : what we've actually told the gimbal (TX thread)

    Each TX tick, current_* is stepped toward wished_* by at most step_per_tick. This
    rate-limits the transmitted setpoint so the gimbal can physically execute each
    command before the next one arrives. The integrator in the inference thread can
    set wished_* freely; current_* will catch up at the gimbal's physical slew rate.

    Threading model:
      - inference_thread calls set_wished() each frame
      - gimbal_tx_thread (driven by run_tx_loop) wakes up at send_interval, ramps
        current_* toward wished_* by at most step_per_tick, and transmits.
    """

    def __init__(self, master, send_rate_hz: float, heartbeat_send_rate_hz: float,
                 max_slew_rate_yaw_dps: float, max_slew_rate_pitch_dps: float,
                 initial_pitch: float, initial_yaw: float, dry_run: bool):
        self.master = master
        self.dry_run = dry_run
        self.send_interval = 1.0 / float(send_rate_hz)
        self.heartbeat_send_interval = 1.0 / float(heartbeat_send_rate_hz)
        self.step_per_tick_yaw = float(max_slew_rate_yaw_dps) * self.send_interval
        self.step_per_tick_pitch = float(max_slew_rate_pitch_dps) * self.send_interval

        self._lock = threading.Lock()
        self._pitch = float(initial_pitch)
        self._yaw = float(initial_yaw)
        self._wished_pitch = float(initial_pitch)
        self._wished_yaw = float(initial_yaw)

        self.send_count = 0
        self.last_send_time = None
        self.last_send_err = None

        # Latest RELAY_STATUS (msg 376) captured by run_rx_loop.
        self._relay_on_mask = 0
        self._relay_present_mask = 0
        self._relay_status_time = None  # time.monotonic() of last update; None=never

        # Latest COMMAND_ACK for DO_REPEAT_RELAY (cmd 182) captured by run_rx_loop.
        # result codes: 0=ACCEPTED, 2=DENIED, 4=FAILED, 5=UNSUPPORTED, 6=IN_PROGRESS
        self._last_repeat_relay_ack_result = None
        self._last_repeat_relay_ack_time = None

    def get_relay_status(self):
        """Returns (on_mask, present_mask, age_seconds_or_None)."""
        with self._lock:
            if self._relay_status_time is None:
                return self._relay_on_mask, self._relay_present_mask, None
            return (self._relay_on_mask, self._relay_present_mask,
                    time.monotonic() - self._relay_status_time)

    def get_last_repeat_relay_ack(self):
        """Returns (result, monotonic_timestamp_or_None). result is None if no ACK
        has ever been observed. Use to verify the autopilot accepted the command."""
        with self._lock:
            return self._last_repeat_relay_ack_result, self._last_repeat_relay_ack_time

    def set_wished(self, pitch: float, yaw: float) -> None:
        with self._lock:
            self._wished_pitch = float(pitch)
            self._wished_yaw = float(yaw)

    def get_current(self) -> tuple:
        with self._lock:
            return self._pitch, self._yaw

    def get_wished(self) -> tuple:
        with self._lock:
            return self._wished_pitch, self._wished_yaw

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
        with self._lock:
            self._transmit(self._pitch, self._yaw)
        last_send = time.monotonic()
        last_heartbeat_send = last_send

        while not stop_event.is_set():
            now = time.monotonic()
            if (now - last_send) >= self.send_interval:
                with self._lock:
                    dy = self._wished_yaw - self._yaw
                    dp = self._wished_pitch - self._pitch
                    if dy > self.step_per_tick_yaw:
                        dy = self.step_per_tick_yaw
                    elif dy < -self.step_per_tick_yaw:
                        dy = -self.step_per_tick_yaw
                    if dp > self.step_per_tick_pitch:
                        dp = self.step_per_tick_pitch
                    elif dp < -self.step_per_tick_pitch:
                        dp = -self.step_per_tick_pitch
                    self._yaw += dy
                    self._pitch += dp
                    moved = abs(dy) > 1e-6 or abs(dp) > 1e-6
                    pitch_now = self._pitch
                    yaw_now = self._yaw

                if moved or (now - last_heartbeat_send) >= self.heartbeat_send_interval:
                    self._transmit(pitch_now, yaw_now)
                    last_send = now
                    last_heartbeat_send = now
                else:
                    last_send = now

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
                    cmd = int(msg.command)
                    if cmd == MAV_CMD_DO_MOUNT_CONTROL:
                        print(f"[ACK] cmd={cmd} result={msg.result}", flush=True)
                    elif cmd == MAV_CMD_DO_SET_RELAY or cmd == MAV_CMD_DO_REPEAT_RELAY:
                        print(f"[ACK-RELAY] cmd={cmd} result={msg.result} "
                              f"(0=ACCEPTED, 4=FAILED, 2=DENIED, 5=UNSUPPORTED)", flush=True)
                        if cmd == MAV_CMD_DO_REPEAT_RELAY:
                            with self._lock:
                                self._last_repeat_relay_ack_result = int(msg.result)
                                self._last_repeat_relay_ack_time = time.monotonic()
                except Exception:
                    pass
            elif t == "RELAY_STATUS":
                try:
                    with self._lock:
                        self._relay_on_mask = int(getattr(msg, "on", 0))
                        self._relay_present_mask = int(getattr(msg, "present", 0))
                        self._relay_status_time = time.monotonic()
                except Exception:
                    pass
            elif t == "STATUSTEXT":
                text = (getattr(msg, "text", "") or "").strip()
                if text:
                    print(f"[STATUSTEXT sev={getattr(msg, 'severity', '?')}] {text}", flush=True)


def read_current_gimbal_position(master, timeout: float) -> tuple:
    """Drain MAVLink inbox briefly looking for MOUNT_STATUS or
    GIMBAL_DEVICE_ATTITUDE_STATUS. Returns (pitch_deg, yaw_deg) or None.

    Used by --start-from-current so the controller's initial reference matches the
    gimbal's actual physical orientation rather than slewing it to (0,0) first.
    """
    if master is None:
        return None
    import math

    end = time.monotonic() + float(timeout)
    while time.monotonic() < end:
        remaining = max(0.05, end - time.monotonic())
        try:
            msg = master.recv_match(blocking=True, timeout=remaining)
        except Exception as e:
            print(f"[MAVLINK] recv_match failed during initial read: {e}", flush=True)
            return None
        if msg is None or msg.get_type() == "BAD_DATA":
            continue
        t = msg.get_type()
        if t == "MOUNT_STATUS":
            try:
                pitch_cdeg = float(getattr(msg, "pointing_a", 0))
                yaw_cdeg = float(getattr(msg, "pointing_c", 0))
                return pitch_cdeg / 100.0, yaw_cdeg / 100.0
            except Exception:
                continue
        if t == "GIMBAL_DEVICE_ATTITUDE_STATUS":
            try:
                q = list(getattr(msg, "q", []))
                if len(q) != 4:
                    continue
                w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                sinp = 2.0 * (w * y - z * x)
                sinp = max(-1.0, min(1.0, sinp))
                pitch_rad = math.asin(sinp)
                yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                return math.degrees(pitch_rad), math.degrees(yaw_rad)
            except Exception:
                continue
    return None


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
        try:
            set_message_interval(master, MAVLINK_MSG_ID_RELAY_STATUS, 200_000)
            print(f"[MAVLINK] Requested RELAY_STATUS (msg {MAVLINK_MSG_ID_RELAY_STATUS}) "
                  f"@ 5 Hz for closed-loop relay verification", flush=True)
        except Exception as e:
            print(f"[MAVLINK] WARNING: SET_MESSAGE_INTERVAL for RELAY_STATUS failed: {e}",
                  flush=True)
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
                        help="Yaw gain — same as simulation script. Controller computes "
                             "wished_yaw = current_yaw + yaw_gain * err_x each frame. "
                             "The TX thread ramps the actual transmitted setpoint toward "
                             "wished_yaw at --max-slew-rate-yaw deg/sec.")
    parser.add_argument("--pitch-gain", type=float, default=10.0,
                        help="Pitch gain — same as simulation script. wished_pitch = "
                             "current_pitch - pitch_gain * err_y. TX ramps toward it.")
    parser.add_argument("--max-slew-rate-yaw", type=float, default=2.0,
                        help="Maximum yaw slew rate (deg/sec) for the TRANSMITTED setpoint. "
                             "Default 2 deg/sec -> 0.10 deg per tick at 20 Hz send rate. "
                             "Very gentle micro-stepping. Raise to 5-10 if convergence is too slow.")
    parser.add_argument("--max-slew-rate-pitch", type=float, default=1.5,
                        help="Maximum pitch slew rate (deg/sec) for the TRANSMITTED setpoint. "
                             "Default 1.5 deg/sec -> 0.075 deg per tick at 20 Hz. "
                             "Raise to 4-8 if too slow.")
    parser.add_argument("--start-from-current", action="store_true",
                        help="At startup, read the gimbal's actual orientation from MOUNT_STATUS "
                             "or GIMBAL_DEVICE_ATTITUDE_STATUS and use that as the initial "
                             "reference. If neither message arrives within --read-current-timeout, "
                             "the script aborts. Without this flag the script CENTERS the gimbal "
                             "(sends (0,0)) at startup before tracking begins.")
    parser.add_argument("--read-current-timeout", type=float, default=2.0,
                        help="Seconds to wait for MOUNT_STATUS / GIMBAL_DEVICE_ATTITUDE_STATUS "
                             "when --start-from-current is set (default: 2.0).")
    parser.add_argument("--initial-pitch", type=float, default=0.0,
                        help="Initial pitch (deg) if not using --start-from-current.")
    parser.add_argument("--initial-yaw", type=float, default=0.0,
                        help="Initial yaw (deg) if not using --start-from-current.")

    parser.add_argument("--no-fire", action="store_true",
                        help="Disable all firing logic. The script will track but never log "
                             "fire events nor send relay commands. Use for camera/gimbal-only "
                             "testing without any weapon engagement.")
    parser.add_argument("--live-fire", action="store_true",
                        help="DANGEROUS. Send a real MAV_CMD_DO_REPEAT_RELAY per centering event. "
                             "The autopilot pulses the relay ON for --fire-period seconds then "
                             "auto-OFFs (cycle's second half is another --fire-period seconds idle). "
                             "RELAY_STATUS gates each phase transition. Without this flag the script "
                             "runs in BLANK mode (logs phase transitions, no relay commands).")
    parser.add_argument("--fire-relay", type=int, default=1,
                        help="Relay instance for DO_REPEAT_RELAY param1 (default: 1, matches the "
                             "QGroundControl 'Shoot Gun' relay).")
    parser.add_argument("--fire-period", type=float, default=5.0,
                        help="Seconds the autopilot holds the relay ON per burst. The script sends "
                             "DO_REPEAT_RELAY with cycles=1, period=2*fire_period; ArduPilot then "
                             "auto-toggles relay ON for fire_period sec, then OFF for fire_period sec. "
                             "Default 5.0 (= cycle_time 10s = 5s ON + 5s OFF).")
    parser.add_argument("--fire-cooldown", type=float, default=0.5,
                        help="Additional seconds of forced idle AFTER the autopilot's full cycle "
                             "(2*fire_period) completes, before the next burst can be triggered. "
                             "Default 0.5. Total time between fires = 2*fire_period + fire_cooldown.")

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
    if args.max_slew_rate_yaw <= 0 or args.max_slew_rate_pitch <= 0:
        parser.error("--max-slew-rate-yaw and --max-slew-rate-pitch must be > 0")
    if args.read_current_timeout <= 0:
        parser.error("--read-current-timeout must be > 0")
    if args.no_fire and args.live_fire:
        parser.error("--no-fire and --live-fire are mutually exclusive")
    if not (0 <= args.fire_relay <= 15):
        parser.error("--fire-relay must be between 0 and 15")

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
    print(f"[INFO] Slew limits: yaw<={args.max_slew_rate_yaw:.1f} deg/sec, "
          f"pitch<={args.max_slew_rate_pitch:.1f} deg/sec", flush=True)

    if args.no_fire:
        fire_mode = "DISABLED"
    elif args.live_fire:
        fire_mode = "LIVE"
    else:
        fire_mode = "BLANK"
    print(f"[INFO] Fire mode: {fire_mode} (relay={args.fire_relay}, "
          f"trigger=centered-in-deadband, continuous)", flush=True)
    if args.live_fire:
        print("[WARN] *** LIVE FIRE ENABLED *** real MAV_CMD_DO_SET_RELAY commands "
              "will be sent. Ctrl-C stops the script and forces relay OFF.", flush=True)

    initial_pitch = clamp(args.initial_pitch, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
    initial_yaw = clamp(args.initial_yaw, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
    if args.start_from_current:
        if master is None:
            print("[INIT] --start-from-current requested but no MAVLink connection; "
                  "falling back to (initial_pitch, initial_yaw).", flush=True)
        else:
            print(f"[INIT] Reading current gimbal orientation (timeout "
                  f"{args.read_current_timeout:.1f}s)...", flush=True)
            pos = read_current_gimbal_position(master, args.read_current_timeout)
            if pos is None:
                print("[INIT] ERROR: no MOUNT_STATUS or GIMBAL_DEVICE_ATTITUDE_STATUS received. "
                      "Aborting (rerun without --start-from-current to center first instead).",
                      flush=True)
                try:
                    master.close()
                except Exception:
                    pass
                sys.exit(3)
            initial_pitch = clamp(pos[0], GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
            initial_yaw = clamp(pos[1], GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
            print(f"[INIT] Read current gimbal: pitch={initial_pitch:+.2f} yaw={initial_yaw:+.2f}",
                  flush=True)
    else:
        if master is not None and not args.no_mavlink:
            print(f"[INIT] Centering gimbal first (pitch=0, yaw=0) for "
                  f"{args.center_duration:.2f}s @ {args.center_rate}Hz...", flush=True)
            center_gimbal(master, args.center_duration, args.center_rate,
                          dry_run=args.no_mavlink or master is None)
        initial_pitch = 0.0
        initial_yaw = 0.0
        print(f"[INIT] Initial reference: pitch={initial_pitch:+.2f} yaw={initial_yaw:+.2f}",
              flush=True)

    link = GimbalLink(
        master=master,
        send_rate_hz=args.send_rate,
        heartbeat_send_rate_hz=args.heartbeat_send_rate,
        max_slew_rate_yaw_dps=args.max_slew_rate_yaw,
        max_slew_rate_pitch_dps=args.max_slew_rate_pitch,
        initial_pitch=initial_pitch,
        initial_yaw=initial_yaw,
        dry_run=args.no_mavlink or master is None,
    )

    latest_frame = [None]
    latest_detections = [[]]
    frame_lock = threading.Lock()
    det_lock = threading.Lock()
    stop_event = threading.Event()
    frame_event = threading.Event()
    frame_seq = [0]

    gimbal_state = {
        "pitch": initial_pitch,
        "yaw": initial_yaw,
        "wished_pitch": initial_pitch,
        "wished_yaw": initial_yaw,
        "frame_idx": 0,
        "firing": False,
        "fire_elapsed": 0.0,
    }
    gimbal_state_lock = threading.Lock()

    # Fire state machine — DO_REPEAT_RELAY (cmd 182) with COMMAND_ACK gating.
    #
    #   IDLE    → wait for CENTERED. On entry: send ONE DO_REPEAT_RELAY → ARMING.
    #   ARMING  → wait for COMMAND_ACK for cmd=182. The autopilot ACKs commands
    #             it accepts within ~100ms regardless of relay timing, so this
    #             is a reliable confirmation (unlike RELAY_STATUS which may not
    #             show intermediate ON state during a 1-cycle DO_REPEAT_RELAY).
    #               result=0  → ACCEPTED → FIRING
    #               result≠0  → REJECTED → IDLE (log error)
    #               timeout   → ACK lost → FIRING (proceed, assume command got through)
    #   FIRING  → autopilot is pulsing the relay. Timer-based for fire_period sec.
    #             RELAY_STATUS read for log visibility only.
    #   COOLDOWN→ wait until autopilot's full cycle (2*fire_period from send)
    #             completes plus fire_cooldown; ensures next DO_REPEAT_RELAY can't
    #             collide with an in-progress cycle.
    #
    # MAVLink traffic per fire: exactly ONE DO_REPEAT_RELAY out, one COMMAND_ACK in.
    # Plus the inbound RELAY_STATUS stream (autopilot publishes at 5 Hz, we just listen).
    fire_burst_duration = float(args.fire_period)
    fire_cycle_time = 2.0 * float(args.fire_period)  # DO_REPEAT_RELAY param3
    fire_arm_ack_timeout = 1.0  # max time to wait for COMMAND_ACK before assuming lost
    fire_state = {
        "phase": "IDLE",
        "fire_send_time": 0.0,
    }

    def _enter_phase(new_phase: str, log: str) -> None:
        fire_state["phase"] = new_phase
        print(log, flush=True)

    def fire_advance(centered: bool):
        """Drive the fire state machine. Called every inference frame."""
        if args.no_fire:
            return
        now = time.monotonic()
        phase = fire_state["phase"]
        mode = "LIVE" if args.live_fire else "BLANK"

        # RELAY_STATUS read for log visibility only.
        on_mask, _, status_age = link.get_relay_status()
        have_fresh_status = status_age is not None and status_age < 2.0
        actual_on = ((on_mask >> args.fire_relay) & 1) if have_fresh_status else None
        relay_tag = f"RELAY[{args.fire_relay}]={'ON' if actual_on == 1 else 'OFF' if actual_on == 0 else '?'}"

        if phase == "IDLE":
            if centered:
                fire_state["fire_send_time"] = now
                if args.live_fire and master is not None:
                    try:
                        send_repeat_relay(master, args.fire_relay, 1, fire_cycle_time)
                    except Exception as e:
                        print(f"[FIRE ERROR] DO_REPEAT_RELAY failed: {e}", flush=True)
                        return
                _enter_phase(
                    "ARMING",
                    f"[{mode} ARMING] sent DO_REPEAT_RELAY({args.fire_relay},cycles=1,"
                    f"period={fire_cycle_time:.2f}s); awaiting COMMAND_ACK for cmd=182",
                )

        elif phase == "ARMING":
            elapsed = now - fire_state["fire_send_time"]
            # BLANK mode auto-confirms after a tick so the state machine progresses
            # without real MAVLink traffic.
            if not args.live_fire:
                if elapsed >= 0.1:
                    _enter_phase(
                        "FIRING",
                        f"[{mode} ARMED] BLANK simulated ACK after {elapsed:.2f}s; "
                        f"autopilot would pulse relay ON for ~{fire_burst_duration:.2f}s",
                    )
                return
            # Check for an ACK that arrived after we sent.
            ack_result, ack_time = link.get_last_repeat_relay_ack()
            ack_is_for_our_send = (
                ack_time is not None and ack_time >= fire_state["fire_send_time"]
            )
            if ack_is_for_our_send:
                if ack_result == 0:
                    _enter_phase(
                        "FIRING",
                        f"[{mode} ARMED] COMMAND_ACK result=0 (ACCEPTED) after "
                        f"{elapsed:.2f}s; autopilot will pulse relay ON for "
                        f"~{fire_burst_duration:.2f}s",
                    )
                else:
                    # Rejected — abort. Don't enter COOLDOWN because the autopilot
                    # didn't start a cycle, so there's nothing to wait for.
                    _enter_phase(
                        "IDLE",
                        f"[{mode} ARMING REJECTED] COMMAND_ACK result={ack_result} "
                        f"(2=DENIED, 4=FAILED, 5=UNSUPPORTED) after {elapsed:.2f}s; "
                        f"NOT firing; back to IDLE",
                    )
            elif elapsed >= fire_arm_ack_timeout:
                # No ACK arrived within the timeout. The packet may have been lost,
                # or the ACK may have been lost. Either way, proceed to FIRING and
                # let the timer/COOLDOWN handle it. Log a warning so the user notices.
                _enter_phase(
                    "FIRING",
                    f"[{mode} ARMED WARN] no COMMAND_ACK in {fire_arm_ack_timeout:.2f}s; "
                    f"assuming command got through (ACK may have been lost). "
                    f"If no fire occurred, autopilot may have rejected silently.",
                )

        elif phase == "FIRING":
            elapsed = now - fire_state["fire_send_time"]
            if elapsed >= fire_burst_duration:
                _enter_phase(
                    "COOLDOWN",
                    f"[{mode} BURST DONE] {fire_burst_duration:.2f}s ON elapsed; "
                    f"{relay_tag}; awaiting cycle completion + {args.fire_cooldown:.2f}s cooldown",
                )

        elif phase == "COOLDOWN":
            elapsed = now - fire_state["fire_send_time"]
            ready_at = fire_cycle_time + float(args.fire_cooldown)
            if elapsed >= ready_at:
                _enter_phase(
                    "IDLE",
                    f"[{mode} FIRE READY] cycle ({fire_cycle_time:.2f}s) + cooldown "
                    f"({args.fire_cooldown:.2f}s) complete; {relay_tag}; re-armed",
                )

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

                # Read where the gimbal actually IS (the TX-thread-ramped commanded position).
                # Recompute wished from that — NOT from a free-running integrator. The wished
                # leads `current` by gain*err; as the gimbal physically slews and err shrinks,
                # wished and current converge. No accumulation, no overshoot.
                cur_pitch, cur_yaw = link.get_current()

                if move_label != "CENTERED":
                    wished_yaw = clamp(cur_yaw + args.yaw_gain * err_x,
                                        GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
                    wished_pitch = clamp(cur_pitch - args.pitch_gain * err_y,
                                          GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
                    link.set_wished(wished_pitch, wished_yaw)
                else:
                    wished_pitch, wished_yaw = link.get_wished()

                # Drive the fire state machine — simple 3-phase timer-driven.
                fire_advance(centered=(move_label == "CENTERED"))

                phase = fire_state["phase"]
                is_firing = (phase == "FIRING")
                fire_elapsed = (time.monotonic() - fire_state["fire_send_time"]) if is_firing else 0.0

                with gimbal_state_lock:
                    gimbal_state["pitch"] = cur_pitch
                    gimbal_state["yaw"] = cur_yaw
                    gimbal_state["wished_pitch"] = wished_pitch
                    gimbal_state["wished_yaw"] = wished_yaw
                    gimbal_state["frame_idx"] = gimbal_frame_idx
                    gimbal_state["firing"] = is_firing
                    gimbal_state["fire_elapsed"] = fire_elapsed

                fire_mode_tag = "LIVE" if args.live_fire else "BLANK"
                if is_firing:
                    fire_info = f"FIRING_{fire_mode_tag}_t={fire_elapsed:.2f}s"
                else:
                    fire_info = f"fire_phase={phase}"

                confidence = float(selected.get("confidence", 0.0))
                print(
                    f"[F{gimbal_frame_idx:06d}] target_bbox=(({int(x1)},{int(y1)}),({int(x2)},{int(y2)})) "
                    f"center=({int(round(cx))},{int(round(cy))}) "
                    f"err=({err_x:+.3f},{err_y:+.3f}) conf={confidence:.3f} dir={move_label} "
                    f"cur=(y{cur_yaw:+.2f},p{cur_pitch:+.2f}) "
                    f"wished=(y{wished_yaw:+.2f},p{wished_pitch:+.2f}) {fire_info} "
                    f"CMD_LONG cmd={MAV_CMD_DO_MOUNT_CONTROL} "
                    f"param1={cur_pitch:.2f} param2=0.00 param3={cur_yaw:.2f} "
                    f"param4=0.00 param5=0.00 param6=0.00 "
                    f"param7={MAV_MOUNT_MODE_MAVLINK_TARGETING}",
                    flush=True,
                )
            else:
                cur_pitch, cur_yaw = link.get_current()
                # Lost target — drive the state machine with centered=False. With
                # the simple timer-driven fire model, this means: don't initiate a
                # new fire while not centered. An in-progress FIRING/COOLDOWN
                # continues to completion regardless (the autopilot owns the cycle).
                fire_advance(centered=False)
                phase = fire_state["phase"]
                is_firing = (phase == "FIRING")
                fire_elapsed = (time.monotonic() - fire_state["fire_send_time"]) if is_firing else 0.0
                with gimbal_state_lock:
                    gimbal_state["pitch"] = cur_pitch
                    gimbal_state["yaw"] = cur_yaw
                    gimbal_state["frame_idx"] = gimbal_frame_idx
                    gimbal_state["firing"] = is_firing
                    gimbal_state["fire_elapsed"] = fire_elapsed
                print(f"[F{gimbal_frame_idx:06d}] NO_TARGET hold "
                      f"cur=(y{cur_yaw:+.2f},p{cur_pitch:+.2f}) fire_phase={phase}",
                      flush=True)

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
                        firing_now = gimbal_state.get("firing", False)
                        fire_elapsed = gimbal_state.get("fire_elapsed", 0.0)
                    if selected is not None:
                        (x1, y1), (x2, y2) = selected["bbox"]
                        tcx = int(round((x1 + x2) / 2.0))
                        tcy = int(round((y1 + y2) / 2.0))
                        primary = (0, 0, 255) if not firing_now else (0, 255, 0)
                        cv2.circle(frame, (tcx, tcy), 5, primary, -1)
                        cv2.line(frame, (fx, fy), (tcx, tcy), primary, 2)
                        cv2.putText(frame, f"yaw={gy:+.1f} pitch={gp:+.1f}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    primary, 2, cv2.LINE_AA)
                        if firing_now:
                            fire_mode_tag = "LIVE" if args.live_fire else "BLANK"
                            cv2.putText(frame, f"FIRING ({fire_mode_tag}) t={fire_elapsed:.2f}s",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 255, 0), 3, cv2.LINE_AA)
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
        if not args.no_fire and master is not None:
            try:
                send_relay(master, args.fire_relay, 0)
                print(f"[FIRE] safety: relay {args.fire_relay} forced OFF on exit", flush=True)
            except Exception as e:
                print(f"[FIRE] WARN: safety relay OFF failed on exit: {e}", flush=True)
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
