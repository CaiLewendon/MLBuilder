"""
tf_live_inferenceV2_drone_auto.py

Production live-inference + autonomous drone-positioning control for the flight
computer. Issues real MAVLink SET_POSITION_TARGET_LOCAL_NED velocity setpoints
to an ArduPilot Copter in GUIDED mode.

Pulls from three pre-existing scripts:
  - test/tf_live_inferenceV2.py                 -> capture/inference/writer pipeline
  - test/tf_live_inferenceV2_gimbal_auto.py     -> 4-thread design + MAVLink helpers
  - test/tf_live_infrence_drone_simulation.py   -> 7-state control flow + OSD

Threads
  1. capture_thread     : RTSP -> latest_frame  (drop-old, single slot)
  2. inference_thread   : latest_frame -> detections + drone state machine
                          + link.set_velocity(...)
  3. drone_tx_thread    : SET_POSITION_TARGET_LOCAL_NED at --tx-rate Hz
  4. drone_rx_thread    : capture HEARTBEAT, DISTANCE_SENSOR, VFR_HUD,
                          LOCAL_POSITION_NED, COMMAND_ACK, STATUSTEXT
  + main thread         : optional video writer (--no-output to skip)

Operating modes (mirrors gimbal_auto's --live-fire pattern):
  - DEFAULT (no --live-fly)  -> BLANK mode: connects to MAVLink, reads telemetry,
                                logs what it WOULD send. Safe for ground testing
                                with autopilot connected but drone disarmed.
  - --live-fly               -> LIVE mode: actually sends velocity setpoints to
                                the autopilot. ONLY operates when autopilot is in
                                GUIDED mode at startup.
  - --no-mavlink             -> pure dry-run, no MAVLink at all, simulated LiDAR
                                from the original drone_simulation script.

GUIDED-mode gate:
  At startup the script checks master.flightmode. If not "GUIDED", prints a clear
  error and exits with code 4. The user must manually switch to GUIDED via QGC
  or the transmitter. The script does NOT send MAV_CMD_DO_SET_MODE.

  During operation, the rx thread monitors HEARTBEAT and logs [MODE LOST] if the
  mode changes out of GUIDED. The tx thread checks before each send.

Pi command (recommended after --start-from-current works on the airframe):
  python3 -B tf_live_inferenceV2_drone_auto.py ~/FullDataSetProd_edgetpu.tflite \
      --tpu -p --no-output --mavlink tcp:10.42.0.1:5760

  (Then arm the drone, switch to GUIDED, then add --live-fly for actual flight.)
"""

import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 10
WIDTH = 1920
HEIGHT = 1080

# MAVLink constants
MAVLINK_MSG_ID_HEARTBEAT = 0
MAVLINK_MSG_ID_LOCAL_POSITION_NED = 32
MAVLINK_MSG_ID_VFR_HUD = 74
MAVLINK_MSG_ID_DISTANCE_SENSOR = 132
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAV_CMD_REQUEST_MESSAGE = 512

# SET_POSITION_TARGET_LOCAL_NED coordinate frames and type masks
MAV_FRAME_BODY_NED = 8        # Body frame, NED axes (forward/right/down)
MAV_FRAME_BODY_OFFSET_NED = 9 # alternative; we use BODY_NED for velocity setpoints

# SET_POSITION_TARGET_LOCAL_NED type_mask bits — set to IGNORE that field.
# We want to keep velocity (3,4,5) and yaw_rate (11), ignore everything else.
#   bit  0 (0x001) = ignore position x
#   bit  1 (0x002) = ignore position y
#   bit  2 (0x004) = ignore position z
#   bit  3 (0x008) = ignore velocity x   <- KEEP this bit CLEAR
#   bit  4 (0x010) = ignore velocity y   <- KEEP this bit CLEAR
#   bit  5 (0x020) = ignore velocity z   <- KEEP this bit CLEAR
#   bit  6 (0x040) = ignore acceleration x
#   bit  7 (0x080) = ignore acceleration y
#   bit  8 (0x100) = ignore acceleration z
#   bit  9 (0x200) = ignore force-set flag
#   bit 10 (0x400) = ignore yaw           <- IGNORE (we don't command yaw position)
#   bit 11 (0x800) = ignore yaw_rate      <- KEEP this bit CLEAR
# Result: ignore pos+accel+force+yaw = 0x001|0x002|0x004|0x040|0x080|0x100|0x200|0x400 = 0x7C7
TYPE_MASK_VELOCITY_YAW_RATE = 0x7C7

# Control state names (verbatim from drone_simulation)
STATE_NO_TARGET = "NO_TARGET"
STATE_CENTERING = "CENTERING"
STATE_APPROACH = "APPROACH"
STATE_HOLD = "HOLD"
STATE_LOCKED = "LOCKED_HOLD"
STATE_ALTITUDE_ADJUST = "ALTITUDE_ADJUST"
STATE_FINAL_HOLD = "FINAL_HOLD"

# Exit codes
EXIT_OK = 0
EXIT_CAMERA_FAIL = 1
EXIT_PROCESS_REQUIRED = 2
EXIT_GUIDED_CHECK_FAILED = 4
EXIT_DISTANCE_SENSOR_TIMEOUT = 5
EXIT_ARG_CONFLICT = 6

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


def yaw_direction_label(err_x: float, deadband: float) -> str:
    if err_x > deadband:
        return "YAW_RIGHT"
    if err_x < -deadband:
        return "YAW_LEFT"
    return "ALIGNED"


def distance_vx_command(
    current_distance_cm: float,
    target_distance_cm: float,
    gain: float,
    max_vx: float,
    min_vx: float,
) -> float:
    """Computes a body-X velocity command from a distance error (cm).
    Returns positive vx (forward) when we're too far from the target,
    negative vx (backward) when too close. Always at least |min_vx|
    in magnitude when correcting (so micro-corrections are visible)."""
    distance_error = current_distance_cm - target_distance_cm
    if distance_error == 0.0:
        return 0.0
    normalized_error = abs(distance_error) / max(1.0, target_distance_cm)
    vx_cmd = clamp(gain * normalized_error, 0.0, max_vx)
    if vx_cmd > 0.0:
        vx_cmd = max(vx_cmd, min_vx)
    return vx_cmd if distance_error > 0.0 else -vx_cmd


def send_set_position_target_local_ned(
    master,
    vx: float,
    vy: float,
    vz: float,
    yaw_rate_rad: float,
) -> None:
    """Send a body-frame velocity setpoint via SET_POSITION_TARGET_LOCAL_NED (msg 84).
    Position/acceleration/yaw are masked off; only velocity + yaw_rate take effect."""
    master.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms (0 = use autopilot's clock)
        master.target_system,
        master.target_component,
        MAV_FRAME_BODY_NED,
        TYPE_MASK_VELOCITY_YAW_RATE,
        0.0, 0.0, 0.0,         # x, y, z (ignored)
        float(vx), float(vy), float(vz),
        0.0, 0.0, 0.0,         # afx, afy, afz (ignored)
        0.0,                   # yaw (ignored)
        float(yaw_rate_rad),
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
    """Legacy request_data_stream — most autopilots still honor this in addition
    to the per-message SET_MESSAGE_INTERVAL calls."""
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


class MovementLink:
    """Owns the MAVLink connection for drone movement commands.

    State (lock-protected):
      - _vx, _vy, _vz, _yaw_rate_rad   : current velocity setpoint
                                          (set by inference, transmitted by tx loop)
      - _last_heartbeat                : (flightmode_str, base_mode, custom_mode,
                                          system_status, monotonic_time)
      - _last_distance_cm              : (cm, monotonic_time)
      - _last_vfr_hud                  : (groundspeed, airspeed, climb, alt, heading, mono_time)
      - _last_local_position           : (x, y, z, vx, vy, vz, mono_time)

    Threading:
      - inference_thread calls set_velocity(vx, vy, vz, yaw_rate_dps) each frame
      - drone_tx_thread (run_tx_loop) sends SET_POSITION_TARGET_LOCAL_NED at
        send_rate_hz; in dry_run mode it logs instead of sending
      - drone_rx_thread (run_rx_loop) captures telemetry and ACKs

    Mode gating: tx_loop checks self.is_guided() before each send. If not GUIDED,
    skips the send and logs (rate-limited)."""

    def __init__(
        self,
        master,
        send_rate_hz: float,
        dry_run: bool,
        require_guided_to_send: bool = True,
    ):
        self.master = master
        self.dry_run = dry_run
        self.require_guided_to_send = require_guided_to_send
        self.send_interval = 1.0 / max(1.0, float(send_rate_hz))

        self._lock = threading.Lock()
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._yaw_rate_rad = 0.0

        self._last_heartbeat = None  # tuple or None
        self._last_distance_cm = None  # tuple (cm, mono_time) or None
        self._last_vfr_hud = None
        self._last_local_position = None
        self._last_command_ack = None  # (cmd, result, mono_time)

        # Counters / diagnostics
        self.send_count = 0
        self.blank_send_count = 0
        self.last_send_time = None
        self.last_send_err = None
        self.last_mode_lost_log = 0.0  # rate-limit MODE_LOST logging

    # --- setters / getters ---

    def set_velocity(self, vx: float, vy: float, vz: float, yaw_rate_dps: float) -> None:
        with self._lock:
            self._vx = float(vx)
            self._vy = float(vy)
            self._vz = float(vz)
            self._yaw_rate_rad = math.radians(float(yaw_rate_dps))

    def get_velocity(self) -> tuple:
        """Returns (vx, vy, vz, yaw_rate_dps)."""
        with self._lock:
            return self._vx, self._vy, self._vz, math.degrees(self._yaw_rate_rad)

    def get_velocity_raw(self) -> tuple:
        """Returns (vx, vy, vz, yaw_rate_rad)."""
        with self._lock:
            return self._vx, self._vy, self._vz, self._yaw_rate_rad

    def get_mode(self) -> tuple:
        """Returns (flightmode_str_or_None, age_seconds_or_None)."""
        with self._lock:
            if self._last_heartbeat is None:
                return None, None
            mode_str, _, _, _, ts = self._last_heartbeat
            return mode_str, time.monotonic() - ts

    def is_guided(self) -> bool:
        mode, age = self.get_mode()
        if mode is None or age is None:
            return False
        if age > 3.0:
            return False  # stale heartbeat — assume mode unknown
        return mode == "GUIDED"

    def get_distance_cm(self) -> tuple:
        """Returns (cm, age_seconds_or_None). cm is None if never received."""
        with self._lock:
            if self._last_distance_cm is None:
                return None, None
            cm, ts = self._last_distance_cm
            return cm, time.monotonic() - ts

    def get_vfr_hud(self) -> tuple:
        """Returns (groundspeed, airspeed, climb, alt, heading, age) or all-None."""
        with self._lock:
            if self._last_vfr_hud is None:
                return None, None, None, None, None, None
            gs, as_, climb, alt, hdg, ts = self._last_vfr_hud
            return gs, as_, climb, alt, hdg, time.monotonic() - ts

    def get_local_position(self) -> tuple:
        """Returns (x, y, z, vx, vy, vz, age) or all-None."""
        with self._lock:
            if self._last_local_position is None:
                return None, None, None, None, None, None, None
            x, y, z, vx, vy, vz, ts = self._last_local_position
            return x, y, z, vx, vy, vz, time.monotonic() - ts

    # --- transmit ---

    def _transmit(self, vx: float, vy: float, vz: float, yaw_rate_rad: float) -> bool:
        """Returns True if the packet was actually sent, False if BLANK / suppressed."""
        if self.dry_run or self.master is None:
            self.blank_send_count += 1
            self.last_send_time = time.monotonic()
            return False
        if self.require_guided_to_send and not self.is_guided():
            # Rate-limit the MODE_LOST log to once every 2 seconds.
            now = time.monotonic()
            if (now - self.last_mode_lost_log) >= 2.0:
                mode, age = self.get_mode()
                mode_str = mode if mode is not None else "UNKNOWN"
                print(f"[MODE LOST] autopilot mode={mode_str} (not GUIDED); "
                      f"suppressing SET_POSITION_TARGET_LOCAL_NED sends", flush=True)
                self.last_mode_lost_log = now
            return False
        try:
            send_set_position_target_local_ned(self.master, vx, vy, vz, yaw_rate_rad)
            self.last_send_err = None
        except Exception as e:
            self.last_send_err = str(e)
            print(f"[MAVLINK] set_position_target_local_ned send failed: {e}", flush=True)
            return False
        self.last_send_time = time.monotonic()
        self.send_count += 1
        return True

    def run_tx_loop(self, stop_event: threading.Event) -> None:
        """Continuously send the current velocity setpoint at send_rate_hz.

        Unlike DO_MOUNT_CONTROL, ArduPilot expects continuous velocity setpoints
        (≥4 Hz) to maintain GUIDED motion; if you stop sending, the autopilot
        will eventually time out the setpoint and revert to hover."""
        last_send = 0.0
        blank_log_interval = 1.0  # in BLANK, log a representative line at this cadence
        last_blank_log = 0.0

        while not stop_event.is_set():
            now = time.monotonic()
            if (now - last_send) >= self.send_interval:
                with self._lock:
                    vx = self._vx
                    vy = self._vy
                    vz = self._vz
                    yaw_rate_rad = self._yaw_rate_rad
                sent = self._transmit(vx, vy, vz, yaw_rate_rad)
                last_send = now
                # BLANK-mode periodic log so the user can see what we would send.
                if (not sent) and self.dry_run:
                    if (now - last_blank_log) >= blank_log_interval:
                        yaw_rate_dps = math.degrees(yaw_rate_rad)
                        print(f"[BLANK SEND] SET_POSITION_TARGET_LOCAL_NED "
                              f"frame=BODY_NED type_mask=0x{TYPE_MASK_VELOCITY_YAW_RATE:03X} "
                              f"vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} "
                              f"yaw_rate={yaw_rate_dps:+.2f}deg/s "
                              f"({math.degrees(yaw_rate_rad):+.2f}deg/s = {yaw_rate_rad:+.4f}rad/s) "
                              f"blank_total={self.blank_send_count}",
                              flush=True)
                        last_blank_log = now
            time.sleep(0.005)

    def send_halt(self) -> bool:
        """Send a single zero-velocity setpoint to halt the drone immediately.
        Called on exit (regardless of dry_run for consistency with other safety paths)."""
        with self._lock:
            self._vx = 0.0
            self._vy = 0.0
            self._vz = 0.0
            self._yaw_rate_rad = 0.0
        if self.master is None or self.dry_run:
            print(f"[HALT] BLANK or no MAVLink — would send vx=vy=vz=0, yaw_rate=0", flush=True)
            return False
        try:
            send_set_position_target_local_ned(self.master, 0.0, 0.0, 0.0, 0.0)
            print(f"[HALT] sent zero-velocity SET_POSITION_TARGET_LOCAL_NED", flush=True)
            return True
        except Exception as e:
            print(f"[HALT] WARN: zero-velocity send failed: {e}", flush=True)
            return False

    # --- receive ---

    def run_rx_loop(self, stop_event: threading.Event) -> None:
        """Drain the MAVLink inbox. Capture telemetry we care about; silently
        consume the rest so the socket buffer doesn't back up."""
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
            now = time.monotonic()
            if t == "HEARTBEAT":
                try:
                    # pymavlink decodes ArduPilot's custom_mode to a human-readable
                    # string and stores it as master.flightmode after each HEARTBEAT.
                    mode_str = getattr(self.master, "flightmode", None)
                    base_mode = int(getattr(msg, "base_mode", 0))
                    custom_mode = int(getattr(msg, "custom_mode", 0))
                    sys_status = int(getattr(msg, "system_status", 0))
                    with self._lock:
                        prev_mode = self._last_heartbeat[0] if self._last_heartbeat else None
                        self._last_heartbeat = (mode_str, base_mode, custom_mode, sys_status, now)
                    if prev_mode is not None and prev_mode != mode_str:
                        print(f"[MODE CHANGE] {prev_mode} -> {mode_str} "
                              f"(custom_mode={custom_mode})", flush=True)
                except Exception:
                    pass
            elif t == "DISTANCE_SENSOR":
                try:
                    current_cm = float(getattr(msg, "current_distance", 0))
                    with self._lock:
                        self._last_distance_cm = (current_cm, now)
                except Exception:
                    pass
            elif t == "VFR_HUD":
                try:
                    gs = float(getattr(msg, "groundspeed", 0.0))
                    as_ = float(getattr(msg, "airspeed", 0.0))
                    climb = float(getattr(msg, "climb", 0.0))
                    alt = float(getattr(msg, "alt", 0.0))
                    hdg = float(getattr(msg, "heading", 0.0))
                    with self._lock:
                        self._last_vfr_hud = (gs, as_, climb, alt, hdg, now)
                except Exception:
                    pass
            elif t == "LOCAL_POSITION_NED":
                try:
                    x = float(getattr(msg, "x", 0.0))
                    y = float(getattr(msg, "y", 0.0))
                    z = float(getattr(msg, "z", 0.0))
                    vx = float(getattr(msg, "vx", 0.0))
                    vy = float(getattr(msg, "vy", 0.0))
                    vz = float(getattr(msg, "vz", 0.0))
                    with self._lock:
                        self._last_local_position = (x, y, z, vx, vy, vz, now)
                except Exception:
                    pass
            elif t == "COMMAND_ACK":
                try:
                    cmd = int(msg.command)
                    result = int(msg.result)
                    with self._lock:
                        self._last_command_ack = (cmd, result, now)
                    # Log anything we'd care about. SET_POSITION_TARGET_LOCAL_NED is
                    # NOT a command_long, so it never ACKs here; this is for
                    # SET_MESSAGE_INTERVAL and any other one-shot commands we send.
                    if cmd == MAV_CMD_SET_MESSAGE_INTERVAL:
                        print(f"[ACK] SET_MESSAGE_INTERVAL result={result}", flush=True)
                except Exception:
                    pass
            elif t == "STATUSTEXT":
                text = (getattr(msg, "text", "") or "").strip()
                if text:
                    print(f"[STATUSTEXT sev={getattr(msg, 'severity', '?')}] {text}",
                          flush=True)


def open_mavlink_and_check_guided(
    connection_str: str,
    heartbeat_timeout: float,
    stream_rate: int,
    require_guided: bool,
):
    """Connect, wait for heartbeat, verify GUIDED, subscribe telemetry.
    Returns (master, flightmode_str). Raises SystemExit on guided-check failure
    if require_guided=True."""
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[MAVLINK] pymavlink not installed. Re-run with --no-mavlink.", flush=True)
        return None, None

    print(f"[MAVLINK] Connecting: {connection_str}", flush=True)
    try:
        master = mavutil.mavlink_connection(connection_str)
        master.wait_heartbeat(timeout=heartbeat_timeout)
        print(
            f"[MAVLINK] heartbeat sysid={master.target_system} "
            f"compid={master.target_component}",
            flush=True,
        )
    except Exception as e:
        print(f"[MAVLINK] WARNING: connection/heartbeat failed: {e}", flush=True)
        return None, None

    mode = getattr(master, "flightmode", None)
    print(f"[MAVLINK] autopilot flightmode={mode}", flush=True)

    if require_guided and mode != "GUIDED":
        print(
            f"\n[GUIDED CHECK FAILED] Autopilot is in flightmode={mode}.\n"
            f"Switch to GUIDED via QGroundControl or your transmitter, then re-run.\n"
            f"This script does NOT auto-switch modes — that's an operator action.\n",
            flush=True,
        )
        try:
            master.close()
        except Exception:
            pass
        sys.exit(EXIT_GUIDED_CHECK_FAILED)

    # Subscribe to the telemetry we depend on. ArduPilot supports SET_MESSAGE_INTERVAL.
    # Also issue the legacy request_data_stream as a belt-and-suspenders measure.
    request_streams(master, stream_rate)

    for msg_id, label in (
        (MAVLINK_MSG_ID_HEARTBEAT, "HEARTBEAT @ 4 Hz (for mode monitoring)"),
        (MAVLINK_MSG_ID_DISTANCE_SENSOR, "DISTANCE_SENSOR @ 5 Hz (range-to-target)"),
        (MAVLINK_MSG_ID_LOCAL_POSITION_NED, "LOCAL_POSITION_NED @ 5 Hz"),
        (MAVLINK_MSG_ID_VFR_HUD, "VFR_HUD @ 5 Hz (groundspeed/climb feedback)"),
    ):
        # 250000 us = 4 Hz (heartbeat), 200000 us = 5 Hz (others)
        interval_us = 250000 if msg_id == MAVLINK_MSG_ID_HEARTBEAT else 200000
        try:
            set_message_interval(master, msg_id, interval_us)
            print(f"[MAVLINK] Requested {label}", flush=True)
        except Exception as e:
            print(f"[MAVLINK] WARNING: SET_MESSAGE_INTERVAL for msg {msg_id} failed: {e}",
                  flush=True)

    return master, mode


def main():
    parser = argparse.ArgumentParser(
        prog="tflive_drone_auto",
        description="TF Live Inference + autonomous drone-positioning control via "
                    "SET_POSITION_TARGET_LOCAL_NED in GUIDED mode.",
    )

    # --- Model / inference ---
    parser.add_argument("model", nargs="?", help="TFLite model file path", type=str)
    parser.add_argument("--nms", "-n", action="store_false",
                        help="Apply NMS post-processing (default: enabled)")
    parser.add_argument("--tpu", "-t", action="store_true",
                        help="Use Coral EdgeTPU (falls back to CPU if unavailable)")
    parser.add_argument("--confidence", "-c", type=float, default=0.01,
                        help="Raw inference confidence threshold")
    parser.add_argument("--labels", "-l", type=str, default=None)
    parser.add_argument("--process", "-p", action="store_true",
                        help="Enable inference processing (required for movement)")
    parser.add_argument("--overlay", "-o", action="store_true",
                        help="Draw detection labels/boxes on output video")
    parser.add_argument("--no-output", action="store_true",
                        help="Disable video output stream (capture + inference only)")

    # --- Detection post-filtering ---
    parser.add_argument("--min-conf", type=float, default=0.05,
                        help="Post-filter minimum confidence")
    parser.add_argument("--max-area-ratio", type=float, default=0.35,
                        help="Reject boxes larger than this frame-area ratio")
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01,
                        help="Edge margin ratio for edge-touch rejection")
    parser.add_argument("--min-track-confidence", type=float, default=0.35,
                        help="Secondary confidence gate for control state machine")

    # --- Crop passes ---
    parser.add_argument("--center-crop-pass", action="store_true",
                        help="Run second inference pass on a center crop")
    parser.add_argument("--center-crop-ratio", type=float, default=0.5)
    parser.add_argument("--crop-center-x", type=float, default=0.5)
    parser.add_argument("--crop-center-y", type=float, default=0.5)
    parser.add_argument("--second-crop-pass", action="store_true")
    parser.add_argument("--second-crop-ratio", type=float, default=0.45)
    parser.add_argument("--second-crop-center-x", type=float, default=0.55)
    parser.add_argument("--second-crop-center-y", type=float, default=0.35)

    # --- CLAHE preprocessing ---
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)

    # --- MAVLink connection ---
    parser.add_argument("--mavlink", type=str, default="tcp:10.42.0.1:5760",
                        help="pymavlink connection string (default: tcp:10.42.0.1:5760)")
    parser.add_argument("--no-mavlink", action="store_true",
                        help="Skip MAVLink entirely; pure dry-run with simulated LiDAR.")
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0)
    parser.add_argument("--stream-rate", type=int, default=10)

    # --- Live-fly gate (DEFAULT OFF = BLANK) ---
    parser.add_argument("--live-fly", action="store_true",
                        help="DANGEROUS. Send real SET_POSITION_TARGET_LOCAL_NED commands "
                             "to the autopilot. Default (without this flag) is BLANK mode: "
                             "connect to MAVLink, read telemetry, log what we WOULD send.")
    parser.add_argument("--no-guided-check", action="store_true",
                        help="Skip the startup GUIDED-mode check. Use only for ground "
                             "testing with the autopilot disarmed in a non-GUIDED mode "
                             "(e.g., STABILIZE on the bench).")
    parser.add_argument("--tx-rate", type=float, default=10.0,
                        help="SET_POSITION_TARGET_LOCAL_NED send rate (Hz). "
                             "Must be >= 4 Hz to keep ArduPilot from timing out. Default 10.")

    # --- Geometry / control gains (from drone_simulation) ---
    parser.add_argument("--deadband", type=float, default=0.08,
                        help="Normalized deadband around image center")
    parser.add_argument("--camera-hfov-deg", type=float, default=78.0,
                        help="Camera horizontal FOV in degrees")

    # CENTERING gains
    parser.add_argument("--yaw-gain", type=float, default=35.0,
                        help="Yaw-rate gain in deg/s at full-scale horizontal error")
    parser.add_argument("--max-yaw-rate", type=float, default=25.0,
                        help="Maximum absolute yaw-rate command (deg/s)")

    # APPROACH gains
    parser.add_argument("--forward-gain", type=float, default=0.80,
                        help="Distance-control gain for vx command in APPROACH")
    parser.add_argument("--max-vx", type=float, default=0.55,
                        help="Maximum forward velocity (m/s)")
    parser.add_argument("--min-distance-correct-vx", type=float, default=0.08,
                        help="Minimum |vx| when distance outside tolerance")
    parser.add_argument("--target-distance-cm", type=float, default=200.0,
                        help="Target stand-off distance in cm")
    parser.add_argument("--distance-tolerance-cm", type=float, default=15.0,
                        help="Allowed stand-off tolerance in cm")

    # HOLD inner-band
    parser.add_argument("--distance-center-band-cm", type=float, default=2.0,
                        help="Tighter inner band around target distance for trim")
    parser.add_argument("--hold-forward-gain", type=float, default=0.35,
                        help="Distance-control gain in HOLD inner band")
    parser.add_argument("--hold-min-distance-correct-vx", type=float, default=0.03,
                        help="Minimum |vx| for trim correction in HOLD")

    # LOCKED_HOLD
    parser.add_argument("--lock-after-approach", action="store_true", default=True,
                        help="Latch lock-state once aligned at target distance")
    parser.add_argument("--no-lock-after-approach", action="store_false",
                        dest="lock_after_approach")
    parser.add_argument("--lock-confirm-frames", type=int, default=8,
                        help="Frames in hold window required before lock engages")
    parser.add_argument("--lock-yaw-gain", type=float, default=45.0,
                        help="Yaw-rate gain during LOCKED_HOLD")
    parser.add_argument("--lock-forward-gain", type=float, default=1.00,
                        help="Distance-control gain during LOCKED_HOLD")
    parser.add_argument("--lock-deadband-scale", type=float, default=0.50,
                        help="Multiplier on base deadband while locked")

    # ALTITUDE_ADJUST & FINAL_HOLD
    parser.add_argument("--altitude-target-y-ratio", type=float, default=0.75,
                        help="Target vertical position in frame (0.75 = bottom quarter)")
    parser.add_argument("--altitude-deadband", type=float, default=0.04,
                        help="Normalized deadband for altitude")
    parser.add_argument("--altitude-gain", type=float, default=0.55,
                        help="Vertical gain for vz")
    parser.add_argument("--max-vz", type=float, default=0.35,
                        help="Maximum vertical velocity (m/s)")
    parser.add_argument("--altitude-lock-confirm-frames", type=int, default=8,
                        help="Frames in altitude deadband before FINAL_HOLD")

    # --- Distance source ---
    parser.add_argument("--simulate-distance", action="store_true",
                        help="Use simulated LiDAR (drift + jitter) instead of real "
                             "DISTANCE_SENSOR. Auto-enabled by --no-mavlink. Otherwise "
                             "the script requires real DISTANCE_SENSOR within "
                             "--distance-sensor-timeout seconds of connect.")
    parser.add_argument("--distance-sensor-timeout", type=float, default=5.0,
                        help="Seconds to wait for first DISTANCE_SENSOR after connect.")

    # --- Simulated-disturbance args (only used when --simulate-distance) ---
    parser.add_argument("--sim-lidar-start-cm", type=float, default=300.0)
    parser.add_argument("--sim-lidar-noise-cm", type=float, default=0.5)
    parser.add_argument("--sim-lidar-min-cm", type=float, default=40.0)
    parser.add_argument("--sim-lidar-max-cm", type=float, default=1000.0)
    parser.add_argument("--sim-lidar-approach-factor", type=float, default=1.0)
    parser.add_argument("--sim-drift-vx-mps", type=float, default=0.03)
    parser.add_argument("--sim-drift-jitter-mps", type=float, default=0.06)
    parser.add_argument("--sim-altitude-response-px-per-m", type=float, default=180.0)
    parser.add_argument("--sim-altitude-drift-vz-mps", type=float, default=0.02)
    parser.add_argument("--sim-altitude-drift-jitter-vz-mps", type=float, default=0.05)
    parser.add_argument("--sim-altitude-noise-px", type=float, default=0.3)

    args = parser.parse_args()

    # --- Arg validation ---
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
    if args.min_track_confidence < 0.0 or args.min_track_confidence > 1.0:
        parser.error("--min-track-confidence must be in [0.0, 1.0]")
    if args.camera_hfov_deg <= 0.0:
        parser.error("--camera-hfov-deg must be > 0.0")
    if args.altitude_target_y_ratio <= 0.0 or args.altitude_target_y_ratio >= 1.0:
        parser.error("--altitude-target-y-ratio must be > 0 and < 1")
    if args.altitude_deadband < 0.0 or args.altitude_deadband >= 1.0:
        parser.error("--altitude-deadband must be >= 0 and < 1")
    if args.max_vx < 0.0 or args.max_yaw_rate < 0.0 or args.max_vz < 0.0:
        parser.error("--max-vx/--max-yaw-rate/--max-vz must be >= 0.0")
    if args.min_distance_correct_vx < 0.0:
        parser.error("--min-distance-correct-vx must be >= 0.0")
    if args.min_distance_correct_vx > args.max_vx:
        parser.error("--min-distance-correct-vx must be <= --max-vx")
    if args.distance_tolerance_cm <= 0.0:
        parser.error("--distance-tolerance-cm must be > 0.0")
    if args.distance_center_band_cm < 0.0:
        parser.error("--distance-center-band-cm must be >= 0.0")
    if args.distance_center_band_cm > args.distance_tolerance_cm:
        parser.error("--distance-center-band-cm must be <= --distance-tolerance-cm")
    if args.lock_confirm_frames <= 0:
        parser.error("--lock-confirm-frames must be > 0")
    if args.altitude_lock_confirm_frames <= 0:
        parser.error("--altitude-lock-confirm-frames must be > 0")
    if args.tx_rate < 4.0:
        parser.error("--tx-rate must be >= 4 Hz (ArduPilot times out lower rates)")
    if args.live_fly and args.no_mavlink:
        print("[ERROR] --live-fly and --no-mavlink are mutually exclusive.", flush=True)
        sys.exit(EXIT_ARG_CONFLICT)
    if args.no_mavlink:
        args.simulate_distance = True  # forced — no telemetry source available

    # --- Camera ---
    cap = cv2.VideoCapture(pipeline3, cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Error: Could not open video stream.", flush=True)
        sys.exit(EXIT_CAMERA_FAIL)

    # --- Writer ---
    writer = None
    if not args.no_output:
        writer = cv2.VideoWriter(pipeline4, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)
        if not writer.isOpened():
            raise RuntimeError("Failed to open GStreamer VideoWriter")
    else:
        print("[NO-OUTPUT] writer disabled; capture + inference only", flush=True)

    if not args.process:
        print("[ERROR] This script requires --process for autonomous movement. "
              "Use tf_live_inferenceV2.py for raw passthrough.", flush=True)
        sys.exit(EXIT_PROCESS_REQUIRED)

    # --- Labels + Model ---
    try:
        labels = load_labels(args.labels)
    except FileNotFoundError as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)
    from MLBuilder.model.tflite.tflitemodel import TFLiteModel
    m = TFLiteModel(args.model)
    m.allocate(tpu=args.tpu)

    # --- MAVLink + GUIDED mode check ---
    master = None
    flightmode = None
    if not args.no_mavlink:
        master, flightmode = open_mavlink_and_check_guided(
            args.mavlink,
            args.heartbeat_timeout,
            args.stream_rate,
            require_guided=(not args.no_guided_check),
        )
        if master is None:
            print("[ERROR] MAVLink connect failed. Re-run with --no-mavlink for "
                  "pure dry-run, or fix the connection.", flush=True)
            sys.exit(EXIT_ARG_CONFLICT)
    else:
        print("[MAVLINK] Disabled via --no-mavlink. Pure dry-run with simulated LiDAR.",
              flush=True)

    # --- Wait briefly for first DISTANCE_SENSOR (real mode only) ---
    if not args.simulate_distance:
        print(f"[INIT] Waiting up to {args.distance_sensor_timeout:.1f}s for first "
              f"DISTANCE_SENSOR...", flush=True)
        end_t = time.monotonic() + args.distance_sensor_timeout
        got = False
        while time.monotonic() < end_t:
            try:
                msg = master.recv_match(blocking=True, timeout=0.3)
            except Exception:
                msg = None
            if msg is not None and msg.get_type() == "DISTANCE_SENSOR":
                cm = float(getattr(msg, "current_distance", 0))
                print(f"[INIT] DISTANCE_SENSOR seen: {cm:.1f} cm", flush=True)
                got = True
                break
        if not got:
            print(f"[ERROR] No DISTANCE_SENSOR within {args.distance_sensor_timeout:.1f}s.\n"
                  f"  Either configure a rangefinder on the airframe or re-run with "
                  f"--simulate-distance.", flush=True)
            try:
                master.close()
            except Exception:
                pass
            sys.exit(EXIT_DISTANCE_SENSOR_TIMEOUT)

    # --- MovementLink ---
    link = MovementLink(
        master=master,
        send_rate_hz=args.tx_rate,
        dry_run=(args.no_mavlink or not args.live_fly),
        require_guided_to_send=(not args.no_guided_check) and (not args.no_mavlink),
    )

    # --- Print startup banner ---
    print("[INFO] Axis assumptions:", flush=True)
    print("  image: +x=right, +y=down, center=(frame_w/2, frame_h/2)", flush=True)
    print("  yaw control: target right -> yaw clockwise (+), target left -> yaw CCW (-)", flush=True)
    print("  translation: once yaw-aligned, +vx=forward closes distance to target", flush=True)
    print("  altitude: +vz means body-down (descend); -vz means body-up (ascend)", flush=True)
    print(f"[INFO] tx mode: "
          f"{'LIVE-FLY' if args.live_fly else 'BLANK (no real sends)'}", flush=True)
    print(f"[INFO] guided check: "
          f"{'enforced — startup mode=' + str(flightmode) if not args.no_guided_check else 'BYPASSED via --no-guided-check'}",
          flush=True)
    print(f"[INFO] distance source: "
          f"{'SIMULATED (drift+jitter)' if args.simulate_distance else 'DISTANCE_SENSOR (msg 132) from autopilot'}",
          flush=True)
    print(f"[INFO] tx rate: {args.tx_rate:.1f} Hz", flush=True)
    print(f"[INFO] target distance: {args.target_distance_cm:.0f} ± "
          f"{args.distance_tolerance_cm:.0f} cm; alt target y-ratio: "
          f"{args.altitude_target_y_ratio:.2f}", flush=True)
    print(f"[INFO] gains: yaw={args.yaw_gain:.1f}deg/s/full forward={args.forward_gain:.2f} "
          f"alt={args.altitude_gain:.2f}", flush=True)
    if args.live_fly:
        print("[WARN] *** LIVE-FLY ENABLED *** SET_POSITION_TARGET_LOCAL_NED will be sent "
              "to the autopilot at the tx rate. Ensure operator is on RC override.",
              flush=True)

    # --- Shared state for threads ---
    latest_frame = [None]
    latest_detections = [[]]
    frame_lock = threading.Lock()
    det_lock = threading.Lock()
    stop_event = threading.Event()
    frame_event = threading.Event()
    frame_seq = [0]

    # Shared overlay state (inference -> main thread for video write)
    overlay_state = {
        "state": STATE_NO_TARGET,
        "move_label": "IDLE",
        "displacement_label": "CENTERED",
        "err_x": 0.0,
        "err_y": 0.0,
        "vx": 0.0,
        "vz": 0.0,
        "yaw_rate": 0.0,
        "lidar_cm": 0.0,
        "lidar_source": "?",
        "dist_status": "?",
        "dist_error_cm": 0.0,
        "lock_status": "OFF",
        "final_hold_status": "OFF",
        "hold_confirm_count": 0,
        "altitude_confirm_count": 0,
        "lock_age": 0,
        "disturb_vx": 0.0,
        "disturb_vz": 0.0,
        "yaw_to_center_deg": 0.0,
        "yaw_vec_dx_px": 0.0,
        "yaw_vec_mag_px": 0.0,
        "alt_vec_dy_px": 0.0,
        "alt_vec_mag_px": 0.0,
        "body_vec_fwd": 1.0,
        "body_vec_right": 0.0,
        "target_y_px": 0.0,
        "alt_error_norm": 0.0,
        "disp_dx_px": 0.0,
        "disp_dy_px": 0.0,
        "disp_mag_px": 0.0,
        "tcx": 0,
        "tcy": 0,
        "frame_idx": 0,
        "raw_detection_count": 0,
        "actual_vx": None,
        "actual_vz": None,
    }
    overlay_lock = threading.Lock()

    # Simulated LiDAR state (only used when args.simulate_distance)
    sim_state = {
        "lidar_cm": clamp(args.sim_lidar_start_cm, args.sim_lidar_min_cm, args.sim_lidar_max_cm),
        "target_cy": None,
        "last_tick": time.perf_counter(),
    }

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
        frame_idx = 0
        locked_on_target = False
        lock_frame_idx = -1
        hold_confirm_count = 0
        altitude_confirm_count = 0
        final_hold_engaged = False
        sim_target_cy = None  # used both for sticky altitude target after lock AND for simulated altitude dynamics

        while not stop_event.is_set():
            frame_event.wait(timeout=0.1)
            frame_event.clear()

            with frame_lock:
                frame = latest_frame[0]
                seq = frame_seq[0]
            if frame is None or seq == last_seen:
                continue
            last_seen = seq

            now = time.perf_counter()
            dt = clamp(now - sim_state["last_tick"], 0.001, 0.2)
            sim_state["last_tick"] = now

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

            raw_detection_count = len(raw)

            filtered = filter_detections(
                raw, frame,
                min_conf=args.min_conf,
                max_area_ratio=args.max_area_ratio,
                edge_margin_ratio=args.edge_margin_ratio,
            )

            # Secondary confidence gate matches drone_simulation.
            out = [d for d in filtered
                   if float(d.get("confidence", 0.0)) >= args.min_track_confidence]

            count += 1
            if count % 30 == 0:
                print(f"infer_frames={count} dets={len(out)}", flush=True)

            with det_lock:
                latest_detections[0] = out

            frame_idx += 1

            selected = None
            if out:
                selected = max(out, key=lambda d: float(d.get("confidence", 0.0)))

            # --- Get distance ---
            if args.simulate_distance:
                lidar_cm = sim_state["lidar_cm"]
                lidar_source = "sim"
            else:
                cm, age = link.get_distance_cm()
                if cm is None or (age is not None and age > 2.0):
                    # Telemetry stale — fall back to last known value, mark source
                    lidar_cm = (sim_state["lidar_cm"] if cm is None else cm)
                    lidar_source = "stale"
                else:
                    lidar_cm = cm
                    lidar_source = "sensor"
                # Keep sim_state in sync so any later fallback is sensible
                sim_state["lidar_cm"] = lidar_cm

            frame_h, frame_w = frame.shape[:2]
            fx = frame_w / 2.0
            fy = frame_h / 2.0

            # Bounds + targets
            low_bound = args.target_distance_cm - args.distance_tolerance_cm
            high_bound = args.target_distance_cm + args.distance_tolerance_cm
            center_low = args.target_distance_cm - args.distance_center_band_cm
            center_high = args.target_distance_cm + args.distance_center_band_cm
            dist_error_cm = lidar_cm - args.target_distance_cm
            target_y_px = args.altitude_target_y_ratio * float(frame_h)

            # Reset frame-local computed values
            state = STATE_NO_TARGET
            move_label = "IDLE"
            displacement_label = "CENTERED"
            err_x = 0.0
            err_y = 0.0
            vx = 0.0
            yaw_rate = 0.0  # deg/s
            vz = 0.0
            disturbance_vx = 0.0
            disturbance_vz = 0.0
            target_confidence = 0.0
            tcx = int(fx)
            tcy = int(fy)
            yaw_to_center_deg = 0.0
            disp_dx_px = 0.0
            disp_dy_px = 0.0
            disp_mag_px = 0.0
            yaw_vec_dx_px = 0.0
            yaw_vec_mag_px = 0.0
            alt_vec_dy_px = 0.0
            alt_vec_mag_px = 0.0
            body_vec_fwd = 1.0
            body_vec_right = 0.0
            alt_error_norm = 0.0
            control_cy = float(fy)

            if selected is None:
                if locked_on_target:
                    state = STATE_LOCKED
                    move_label = "LOCKED_NO_TARGET"
                    displacement_label = "LOCKED_NO_TARGET"
                    altitude_confirm_count = 0
                    if abs(dist_error_cm) > args.distance_center_band_cm:
                        vx = distance_vx_command(
                            lidar_cm,
                            args.target_distance_cm,
                            args.lock_forward_gain,
                            args.max_vx,
                            args.min_distance_correct_vx,
                        )
                        move_label = "LOCKED_DIST_CORRECT"
                else:
                    state = STATE_NO_TARGET
                    hold_confirm_count = 0
                    altitude_confirm_count = 0
                    final_hold_engaged = False
                    sim_target_cy = None
            else:
                (x1, y1), (x2, y2) = selected["bbox"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if (not locked_on_target) or (sim_target_cy is None):
                    sim_target_cy = float(cy)

                control_cy = float(cy)
                if locked_on_target and sim_target_cy is not None:
                    control_cy = float(sim_target_cy)

                tcx = int(round(cx))
                tcy = int(round(control_cy))
                err_x = (cx - float(fx)) / max(1.0, float(fx))
                err_y = (control_cy - float(fy)) / max(1.0, float(fy))
                yaw_to_center_deg = err_x * (args.camera_hfov_deg * 0.5)
                disp_dx_px = cx - float(fx)
                disp_dy_px = control_cy - float(fy)
                disp_mag_px = math.hypot(disp_dx_px, disp_dy_px)
                yaw_vec_dx_px = disp_dx_px
                yaw_vec_mag_px = abs(yaw_vec_dx_px)
                yaw_rad = math.radians(yaw_to_center_deg)
                body_vec_fwd = math.cos(yaw_rad)
                body_vec_right = math.sin(yaw_rad)
                target_confidence = float(selected.get("confidence", 0.0))
                alt_error_px = target_y_px - control_cy
                alt_error_norm = alt_error_px / max(1.0, float(fy))
                alt_vec_dy_px = alt_error_px
                alt_vec_mag_px = abs(alt_vec_dy_px)

                displacement_label = direction_label(err_x, err_y, args.deadband)
                yaw_label = yaw_direction_label(err_x, args.deadband)
                yaw_aligned = yaw_label == "ALIGNED"
                in_distance_window = low_bound <= lidar_cm <= high_bound

                if locked_on_target:
                    state = STATE_LOCKED
                    lock_deadband = args.deadband * args.lock_deadband_scale
                    move_label = "LOCKED_STEADY"
                    if abs(err_x) > lock_deadband:
                        yaw_rate = clamp(args.lock_yaw_gain * err_x,
                                         -args.max_yaw_rate, args.max_yaw_rate)
                        move_label = "LOCKED_YAW_CORRECT"
                    if abs(dist_error_cm) > args.distance_center_band_cm:
                        vx = distance_vx_command(
                            lidar_cm,
                            args.target_distance_cm,
                            args.lock_forward_gain,
                            args.max_vx,
                            args.min_distance_correct_vx,
                        )
                        if move_label == "LOCKED_STEADY":
                            move_label = "LOCKED_DIST_CORRECT"
                        else:
                            move_label = "LOCKED_YAW+DIST_CORRECT"

                    if not final_hold_engaged:
                        state = STATE_ALTITUDE_ADJUST
                        if abs(alt_error_norm) > args.altitude_deadband:
                            altitude_confirm_count = 0
                            vz = clamp(-args.altitude_gain * alt_error_norm,
                                       -args.max_vz, args.max_vz)
                            if move_label == "LOCKED_STEADY":
                                move_label = "ALT_ADJUST_ONLY"
                            else:
                                move_label = f"{move_label}+ALT_ADJUST"
                        else:
                            altitude_confirm_count += 1
                            move_label = "ALT_CONFIRM"
                            if altitude_confirm_count >= args.altitude_lock_confirm_frames:
                                final_hold_engaged = True
                                state = STATE_FINAL_HOLD
                                move_label = "FINAL_HOLD"
                    else:
                        state = STATE_FINAL_HOLD
                        if abs(alt_error_norm) > args.altitude_deadband:
                            vz = clamp(-args.altitude_gain * alt_error_norm,
                                       -args.max_vz, args.max_vz)
                            if move_label == "LOCKED_STEADY":
                                move_label = "FINAL_HOLD_ALT_CORRECT"
                            else:
                                move_label = f"{move_label}+ALT_HOLD_CORRECT"
                else:
                    final_hold_engaged = False
                    altitude_confirm_count = 0
                    if not yaw_aligned:
                        hold_confirm_count = 0
                        state = STATE_CENTERING
                        yaw_rate = clamp(args.yaw_gain * err_x,
                                         -args.max_yaw_rate, args.max_yaw_rate)
                        move_label = yaw_label
                    else:
                        if lidar_cm > high_bound:
                            hold_confirm_count = 0
                            state = STATE_APPROACH
                            vx = distance_vx_command(
                                lidar_cm,
                                args.target_distance_cm,
                                args.forward_gain,
                                args.max_vx,
                                args.min_distance_correct_vx,
                            )
                            move_label = "FORWARD"
                        elif lidar_cm < low_bound:
                            hold_confirm_count = 0
                            state = STATE_APPROACH
                            vx = distance_vx_command(
                                lidar_cm,
                                args.target_distance_cm,
                                args.forward_gain,
                                args.max_vx,
                                args.min_distance_correct_vx,
                            )
                            move_label = "BACKWARD"
                        else:
                            state = STATE_HOLD
                            move_label = "HOLD"
                            hold_confirm_count += 1
                            if abs(dist_error_cm) > args.distance_center_band_cm:
                                vx = distance_vx_command(
                                    lidar_cm,
                                    args.target_distance_cm,
                                    args.hold_forward_gain,
                                    args.max_vx,
                                    args.hold_min_distance_correct_vx,
                                )
                                if vx > 0.0:
                                    move_label = "HOLD_TRIM_FORWARD"
                                elif vx < 0.0:
                                    move_label = "HOLD_TRIM_BACKWARD"

                    if (
                        args.lock_after_approach
                        and yaw_aligned
                        and in_distance_window
                        and hold_confirm_count >= args.lock_confirm_frames
                    ):
                        locked_on_target = True
                        lock_frame_idx = frame_idx
                        state = STATE_LOCKED
                        move_label = "LOCKED_STEADY"
                        altitude_confirm_count = 0
                        final_hold_engaged = False
                        print(f"[LOCK ENGAGED] hold_frames={hold_confirm_count} "
                              f"lidar={lidar_cm:.1f}cm err_x={err_x:+.3f}", flush=True)

            # --- Publish velocity to the TX thread ---
            link.set_velocity(vx, 0.0, vz, yaw_rate)

            # --- Simulated LiDAR / altitude dynamics (only when simulate-distance) ---
            if args.simulate_distance:
                disturbance_vx = args.sim_drift_vx_mps
                if args.sim_drift_jitter_mps > 0.0:
                    disturbance_vx += random.uniform(
                        -args.sim_drift_jitter_mps, args.sim_drift_jitter_mps)
                net_vx = vx + disturbance_vx
                sim_state["lidar_cm"] += -net_vx * dt * 100.0 * args.sim_lidar_approach_factor
                if args.sim_lidar_noise_cm > 0.0:
                    sim_state["lidar_cm"] += random.uniform(
                        -args.sim_lidar_noise_cm, args.sim_lidar_noise_cm)
                sim_state["lidar_cm"] = clamp(
                    sim_state["lidar_cm"], args.sim_lidar_min_cm, args.sim_lidar_max_cm)

                disturbance_vz = args.sim_altitude_drift_vz_mps
                if args.sim_altitude_drift_jitter_vz_mps > 0.0:
                    disturbance_vz += random.uniform(
                        -args.sim_altitude_drift_jitter_vz_mps,
                        args.sim_altitude_drift_jitter_vz_mps)
                if sim_target_cy is not None:
                    net_vz = vz + disturbance_vz
                    sim_target_cy += -net_vz * dt * args.sim_altitude_response_px_per_m
                    if args.sim_altitude_noise_px > 0.0:
                        sim_target_cy += random.uniform(
                            -args.sim_altitude_noise_px, args.sim_altitude_noise_px)
                    sim_target_cy = clamp(sim_target_cy, 0.0, float(frame_h - 1))

            # --- Distance status (FAR / CLOSE / ON_TARGET / CENTER_BAND / IN_WINDOW_TRIMMING) ---
            if lidar_cm > high_bound:
                dist_status = "FAR"
            elif lidar_cm < low_bound:
                dist_status = "CLOSE"
            elif lidar_cm > center_high or lidar_cm < center_low:
                dist_status = "IN_WINDOW_TRIMMING"
            else:
                dist_status = "CENTER_BAND"

            lock_status = "ON" if locked_on_target else "OFF"
            final_hold_status = "ON" if final_hold_engaged else "OFF"
            lock_age = 0 if lock_frame_idx < 0 else frame_idx - lock_frame_idx

            # --- Real velocity feedback (when telemetry available) ---
            vfr_gs, _, vfr_climb, _, _, vfr_age = link.get_vfr_hud()
            actual_vx = vfr_gs if (vfr_age is not None and vfr_age < 2.0) else None
            actual_vz = vfr_climb if (vfr_age is not None and vfr_age < 2.0) else None
            # ArduPilot VFR_HUD climb is positive UP; convert to body-z down convention
            if actual_vz is not None:
                actual_vz = -actual_vz

            # --- Publish overlay state ---
            with overlay_lock:
                overlay_state["state"] = state
                overlay_state["move_label"] = move_label
                overlay_state["displacement_label"] = displacement_label
                overlay_state["err_x"] = err_x
                overlay_state["err_y"] = err_y
                overlay_state["vx"] = vx
                overlay_state["vz"] = vz
                overlay_state["yaw_rate"] = yaw_rate
                overlay_state["lidar_cm"] = lidar_cm
                overlay_state["lidar_source"] = lidar_source
                overlay_state["dist_status"] = dist_status
                overlay_state["dist_error_cm"] = dist_error_cm
                overlay_state["lock_status"] = lock_status
                overlay_state["final_hold_status"] = final_hold_status
                overlay_state["hold_confirm_count"] = hold_confirm_count
                overlay_state["altitude_confirm_count"] = altitude_confirm_count
                overlay_state["lock_age"] = lock_age
                overlay_state["disturb_vx"] = disturbance_vx
                overlay_state["disturb_vz"] = disturbance_vz
                overlay_state["yaw_to_center_deg"] = yaw_to_center_deg
                overlay_state["yaw_vec_dx_px"] = yaw_vec_dx_px
                overlay_state["yaw_vec_mag_px"] = yaw_vec_mag_px
                overlay_state["alt_vec_dy_px"] = alt_vec_dy_px
                overlay_state["alt_vec_mag_px"] = alt_vec_mag_px
                overlay_state["body_vec_fwd"] = body_vec_fwd
                overlay_state["body_vec_right"] = body_vec_right
                overlay_state["target_y_px"] = target_y_px
                overlay_state["alt_error_norm"] = alt_error_norm
                overlay_state["disp_dx_px"] = disp_dx_px
                overlay_state["disp_dy_px"] = disp_dy_px
                overlay_state["disp_mag_px"] = disp_mag_px
                overlay_state["tcx"] = tcx
                overlay_state["tcy"] = tcy
                overlay_state["frame_idx"] = frame_idx
                overlay_state["raw_detection_count"] = raw_detection_count
                overlay_state["actual_vx"] = actual_vx
                overlay_state["actual_vz"] = actual_vz

            # --- Per-frame log line ---
            mode_str, mode_age = link.get_mode()
            mode_tag = mode_str if mode_str else "NO_HB"
            tx_tag = "LIVE" if args.live_fly else "BLANK"
            if selected is None:
                print(
                    f"[F{frame_idx:06d}] NO_TARGET state={state} "
                    f"mode={mode_tag} tx={tx_tag} "
                    f"lidar={lidar_cm:.1f}cm({lidar_source}) "
                    f"lock={lock_status} hold={hold_confirm_count}/{args.lock_confirm_frames} "
                    f"alt={altitude_confirm_count}/{args.altitude_lock_confirm_frames} "
                    f"final_hold={final_hold_status} "
                    f"CMD vx={vx:+.3f} vz={vz:+.3f} yaw_rate={yaw_rate:+.2f}deg/s",
                    flush=True,
                )
            else:
                (x1, y1), (x2, y2) = selected["bbox"]
                print(
                    f"[F{frame_idx:06d}] target_bbox=(({int(x1)},{int(y1)}),({int(x2)},{int(y2)})) "
                    f"center=({tcx},{tcy}) err=({err_x:+.3f},{err_y:+.3f}) "
                    f"conf={target_confidence:.3f} state={state} action={move_label} "
                    f"mode={mode_tag} tx={tx_tag} "
                    f"lidar={lidar_cm:.1f}cm({lidar_source})/{dist_status} "
                    f"dist_err={dist_error_cm:+.1f}cm "
                    f"lock={lock_status} hold={hold_confirm_count}/{args.lock_confirm_frames} "
                    f"alt={altitude_confirm_count}/{args.altitude_lock_confirm_frames} "
                    f"final_hold={final_hold_status} "
                    f"yaw_to_center={yaw_to_center_deg:+.2f}deg "
                    f"CMD vx={vx:+.3f}m/s vz={vz:+.3f}m/s yaw_rate={yaw_rate:+.2f}deg/s "
                    + (f"actual_vx={actual_vx:.2f}m/s " if actual_vx is not None else "")
                    + (f"actual_vz={actual_vz:+.2f}m/s " if actual_vz is not None else ""),
                    flush=True,
                )

    # --- Thread spawn ---
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

    # --- Main thread: video writer loop with full OSD overlay ---
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
                with overlay_lock:
                    o = dict(overlay_state)

                selected = None
                if detections:
                    selected = max(detections, key=lambda d: float(d.get("confidence", 0.0)))

                if args.overlay:
                    # Detection boxes
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
                        cv2.rectangle(frame, (x_min, y_min - th - 6),
                                      (x_min + tw, y_min), color, -1)
                        cv2.putText(frame, text, (x_min, y_min - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    frame_h, frame_w = frame.shape[:2]
                    fx_i = int(frame_w / 2.0)
                    fy_i = int(frame_h / 2.0)
                    cv2.drawMarker(frame, (fx_i, fy_i), (255, 255, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(frame, "CENTER AXIS", (fx_i + 10, fy_i - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                    # Target + yaw vector
                    if selected is not None:
                        tcx = int(o["tcx"])
                        tcy = int(o["tcy"])
                        cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)
                        yaw_end_x = int(round(float(fx_i) + o["yaw_vec_dx_px"]))
                        cv2.arrowedLine(frame, (fx_i, fy_i), (yaw_end_x, fy_i),
                                        (0, 255, 255), 2, tipLength=0.18)
                        cv2.line(frame, (yaw_end_x, fy_i - 8), (yaw_end_x, fy_i + 8),
                                 (0, 255, 255), 2)
                        # Altitude goal
                        alt_target_y = int(round(o["target_y_px"]))
                        cv2.line(frame, (0, alt_target_y), (frame_w - 1, alt_target_y),
                                 (120, 80, 255), 1)
                        cv2.arrowedLine(frame, (tcx, tcy), (tcx, alt_target_y),
                                        (255, 170, 0), 2, tipLength=0.18)
                        cv2.circle(frame, (tcx, alt_target_y), 5, (255, 170, 0), -1)
                        cv2.putText(frame, "ALT TARGET",
                                    (tcx + 8, max(20, alt_target_y - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 170, 0),
                                    1, cv2.LINE_AA)
                        cv2.putText(frame, "ALT GOAL LINE",
                                    (10, max(20, alt_target_y - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 80, 255),
                                    1, cv2.LINE_AA)

                    # Text status block
                    cv2.putText(
                        frame,
                        f"state={o['state']} action={o['move_label']} disp={o['displacement_label']} "
                        f"err=({o['err_x']:+.3f},{o['err_y']:+.3f})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.putText(
                        frame,
                        f"lidar={o['lidar_cm']:.1f}cm({o['lidar_source']}) "
                        f"({o['dist_status']}) target={args.target_distance_cm:.0f}"
                        f"+-{args.distance_tolerance_cm:.0f} "
                        f"dist_err={o['dist_error_cm']:+.1f}cm "
                        f"cmd[yaw_rate,vx,vz]=({o['yaw_rate']:+.1f}deg/s,"
                        f"{o['vx']:+.2f}m/s,{o['vz']:+.2f}m/s)",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255),
                        1, cv2.LINE_AA)

                    lock_color = ((80, 255, 80) if o["final_hold_status"] == "ON"
                                  else ((120, 230, 255) if o["lock_status"] == "ON"
                                        else (200, 200, 200)))
                    cv2.putText(
                        frame,
                        f"lock={o['lock_status']} "
                        f"hold_frames={o['hold_confirm_count']}/{args.lock_confirm_frames} "
                        f"alt_frames={o['altitude_confirm_count']}/{args.altitude_lock_confirm_frames} "
                        f"final_hold={o['final_hold_status']} lock_age={o['lock_age']} "
                        f"disturb[vx,vz]=({o['disturb_vx']:+.2f},{o['disturb_vz']:+.2f})m/s",
                        (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, lock_color, 2, cv2.LINE_AA)

                    # tx mode + autopilot mode indicator
                    mode_str, mode_age = link.get_mode()
                    mode_display = mode_str if mode_str else "NO_HEARTBEAT"
                    mode_color = (0, 255, 0) if mode_display == "GUIDED" else (0, 120, 255)
                    tx_display = "LIVE-FLY" if args.live_fly else "BLANK"
                    tx_color = (0, 80, 255) if args.live_fly else (200, 200, 0)
                    cv2.putText(
                        frame,
                        f"mode={mode_display}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 2, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        f"tx={tx_display}",
                        (240, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.58, tx_color, 2, cv2.LINE_AA)

                    # Feedback from autopilot (VFR_HUD)
                    if o.get("actual_vx") is not None:
                        cv2.putText(
                            frame,
                            f"feedback: gs={o['actual_vx']:.2f}m/s "
                            f"vz_body={o['actual_vz']:+.2f}m/s",
                            (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (200, 200, 255), 1, cv2.LINE_AA)

                    cv2.putText(
                        frame,
                        f"yaw_to_center={o['yaw_to_center_deg']:+.2f} deg",
                        (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255),
                        2, cv2.LINE_AA)

                    cv2.putText(
                        frame,
                        f"yaw_vec_px=({o['yaw_vec_dx_px']:+.1f},+0.0) "
                        f"|v|={o['yaw_vec_mag_px']:.1f}px "
                        f"bearing_vec_body=(fwd={o['body_vec_fwd']:+.3f},"
                        f"right={o['body_vec_right']:+.3f})",
                        (10, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180),
                        1, cv2.LINE_AA)

                    cv2.putText(
                        frame,
                        f"alt_to_goal_vec_px=(+0.0,{o['alt_vec_dy_px']:+.1f}) "
                        f"|v|={o['alt_vec_mag_px']:.1f}px "
                        f"goal_y={o['target_y_px']:.1f}px "
                        f"alt_err_norm={o['alt_error_norm']:+.3f}",
                        (10, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 120),
                        1, cv2.LINE_AA)

                    cv2.putText(
                        frame,
                        f"target_offset_ref_px=({o['disp_dx_px']:+.1f},"
                        f"{o['disp_dy_px']:+.1f}) |v|={o['disp_mag_px']:.1f}px",
                        (10, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180),
                        1, cv2.LINE_AA)

                    cv2.putText(
                        frame,
                        f"detections={len(detections)} "
                        f"raw={o['raw_detection_count']} frame={o['frame_idx']}",
                        (10, frame_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2, cv2.LINE_AA)

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
        # HALT the drone before disconnecting (only matters in --live-fly)
        if master is not None and args.live_fly:
            try:
                link.send_halt()
            except Exception as e:
                print(f"[HALT] WARN: halt failed on exit: {e}", flush=True)
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
