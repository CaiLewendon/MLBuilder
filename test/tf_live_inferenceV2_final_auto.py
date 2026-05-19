"""
tf_live_inferenceV2_final_auto.py

Big City RPAS Task 2 (Fire Extinguishing) — combined autonomous engagement.
Sequential one-shot mission: drone positions, gimbal aims, water discharges,
photo is captured, autopilot is switched to LOITER, pilot resumes control.

Architecture (single MAVLink master shared by both Link classes):
  Threads
    1. capture_thread       : RTSP -> latest_frame (drop-old, single slot)
    2. inference_thread     : detection + phase-conditional state machines
                              (owns mission_phase transitions)
    3. drone_tx_thread      : SET_POSITION_TARGET_LOCAL_NED at --tx-rate Hz
    4. gimbal_tx_thread     : slew-rate-limited DO_MOUNT_CONTROL at --send-rate Hz
    5. shared_rx_thread     : ONE recv_match loop; dispatches by msg type to
                              drone_link.handle_message / gimbal_link.handle_message
    + main thread           : OSD overlay + optional video writer

Mission flow
  PHASE_DRONE_POSITIONING
      drone-auto 7-state machine runs; gimbal held STATIC at startup angle.
      First-frame compliance check: first valid DISTANCE_SENSOR reading must be
      >= --min-start-distance-cm (Task 2 §5.2.4 requires >2 m approach start).
      When drone state == STATE_FINAL_HOLD: → PHASE_HANDOFF_WAIT.
  PHASE_HANDOFF_WAIT
      Count consecutive FINAL_HOLD frames. Reset to 0 if state regresses.
      When count >= --handoff-confirm-frames: → PHASE_GIMBAL_TRACKING.
  PHASE_GIMBAL_TRACKING
      Drone enters FREEZE (continuous zero-velocity setpoints at --tx-rate).
      Gimbal slew-rate-limited tracking activates.
      Discharge state machine ticks. On CENTERED: IDLE → ARMING (one
      DO_REPEAT_RELAY), → FIRING after COMMAND_ACK.
      When fire phase enters FIRING: → PHASE_FIRING.
  PHASE_FIRING
      Drone still freezing. Discharge timer counts --fire-period seconds.
      When fire phase transitions FIRING → COOLDOWN: → PHASE_VERIFY.
  PHASE_VERIFY
      Drone frozen; gimbal continues to aim at target.
      Captures --capture-frame-count frames at --capture-frame-interval spacing.
      Picks best frame (highest detection confidence; Laplacian-variance fallback).
      Saves <photo-output-dir>/Task_2_<team_name>_target_<#>_<ts>.jpg.
      Prints prominent operator instruction to verify and upload before declaring.
      → PHASE_HANDBACK.
  PHASE_HANDBACK
      Idempotent: send MAV_CMD_DO_SET_MODE → LOITER (or --handback-mode);
      send_halt() for safety; defensive DO_SET_RELAY OFF.
      On DO_SET_MODE ACK (or --handback-mode-timeout): → PHASE_DONE.

Bench (laptop dry-run, simulated LiDAR):
  venv/bin/python test/tf_live_inferenceV2_final_auto.py \\
      export/project1_prod_saved_model/project1_prod_float16.tflite \\
      -p --video 0 --no-mavlink --overlay \\
      --team-name dev_test --target-number 1 --no-photo-capture

Pi BLANK-mode (autopilot connected, drone disarmed in GUIDED):
  python3 -B tf_live_inferenceV2_final_auto.py ~/FullDataSetProd_edgetpu.tflite \\
      --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --sharpen 0.4 \\
      --start-from-current-gimbal --team-name dev_test --target-number 1

Pi LIVE engagement (observer on RC, GUIDED + armed, drone >2 m from target):
  python3 -B tf_live_inferenceV2_final_auto.py ~/FullDataSetProd_edgetpu.tflite \\
      --tpu -p --no-output --mavlink tcp:10.42.0.1:5760 --sharpen 0.4 \\
      --start-from-current-gimbal --live-fly --live-fire \\
      --team-name <team_name> --target-number <N> \\
      --max-vx 0.20 --max-vz 0.15 --max-yaw-rate 10.0 \\
      --target-distance-cm 300 --distance-tolerance-cm 30

Exit codes:
  0  OK
  1  camera or labels failure
  2  --process required for autonomous engagement
  3  --start-from-current-gimbal could not read MOUNT_STATUS in timeout
  4  GUIDED-mode check failed at startup
  5  DISTANCE_SENSOR timeout at startup
  6  CLI flag conflict
  7  Task 2 compliance failure (start distance < --min-start-distance-cm)
"""

import argparse
import math
import pathlib
import random
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 10
WIDTH = 1920
HEIGHT = 1080

# MAVLink message IDs / commands
MAVLINK_MSG_ID_HEARTBEAT = 0
MAVLINK_MSG_ID_LOCAL_POSITION_NED = 32
MAVLINK_MSG_ID_VFR_HUD = 74
MAVLINK_MSG_ID_DISTANCE_SENSOR = 132
MAVLINK_MSG_ID_RELAY_STATUS = 376

MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_SET_RELAY = 181
MAV_CMD_DO_REPEAT_RELAY = 182
MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAV_CMD_REQUEST_MESSAGE = 512

MAV_MOUNT_MODE_MAVLINK_TARGETING = 2

# SET_POSITION_TARGET_LOCAL_NED — body frame, velocity + yaw_rate only
MAV_FRAME_BODY_NED = 8
# bits set = IGNORE that field; we keep velocity (3,4,5) and yaw_rate (11) active.
TYPE_MASK_VELOCITY_YAW_RATE = 0x7C7

# Gimbal physical limits (matches manual_gimbal_control + gimbal_auto)
GIMBAL_YAW_MIN_DEG = -90.0
GIMBAL_YAW_MAX_DEG = 90.0
GIMBAL_PITCH_MIN_DEG = -45.0
GIMBAL_PITCH_MAX_DEG = 45.0

# Drone state-machine names (verbatim from drone_simulation)
STATE_NO_TARGET = "NO_TARGET"
STATE_CENTERING = "CENTERING"
STATE_APPROACH = "APPROACH"
STATE_HOLD = "HOLD"
STATE_LOCKED = "LOCKED_HOLD"
STATE_ALTITUDE_ADJUST = "ALTITUDE_ADJUST"
STATE_FINAL_HOLD = "FINAL_HOLD"

# Mission-phase names
PHASE_DRONE_POSITIONING = "DRONE_POSITIONING"
PHASE_HANDOFF_WAIT = "HANDOFF_WAIT"
PHASE_GIMBAL_TRACKING = "GIMBAL_TRACKING"
PHASE_FIRING = "FIRING"
PHASE_VERIFY = "VERIFY"
PHASE_HANDBACK = "HANDBACK"
PHASE_DONE = "DONE"

# Exit codes
EXIT_OK = 0
EXIT_CAMERA_FAIL = 1
EXIT_PROCESS_REQUIRED = 2
EXIT_GIMBAL_INIT_FAIL = 3
EXIT_GUIDED_CHECK_FAILED = 4
EXIT_DISTANCE_SENSOR_TIMEOUT = 5
EXIT_ARG_CONFLICT = 6
EXIT_COMPLIANCE_FAIL = 7

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


# ============================================================================
# Helpers (verbatim from both source scripts — identical bodies)
# ============================================================================


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


def apply_grayscale_bgr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_luminance_bgr(frame, factor: float):
    if factor == 1.0:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.convertScaleAbs(l, alpha=factor, beta=0)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_contrast_bgr(frame, factor: float):
    if factor == 1.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=factor, beta=128.0 * (1.0 - factor))


def apply_saturation_bgr(frame, factor: float):
    if factor == 1.0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=factor, beta=0)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def apply_unsharp_bgr(frame, amount: float, sigma: float = 1.0):
    if amount <= 0.0:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def apply_preproc(frame, args):
    if args.grayscale:
        frame = apply_grayscale_bgr(frame)
    if args.luminance != 1.0:
        frame = apply_luminance_bgr(frame, args.luminance)
    if args.contrast != 1.0:
        frame = apply_contrast_bgr(frame, args.contrast)
    if args.saturation != 1.0 and not args.grayscale:
        frame = apply_saturation_bgr(frame, args.saturation)
    if args.sharpen > 0.0:
        frame = apply_unsharp_bgr(frame, args.sharpen, args.sharpen_sigma)
    return frame


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
    """Body-X velocity command from a distance error (cm). Forward (+vx) when
    we're too far; backward (-vx) when too close. At least |min_vx| in
    magnitude when correcting so micro-corrections are visible."""
    distance_error = current_distance_cm - target_distance_cm
    if distance_error == 0.0:
        return 0.0
    normalized_error = abs(distance_error) / max(1.0, target_distance_cm)
    vx_cmd = clamp(gain * normalized_error, 0.0, max_vx)
    if vx_cmd > 0.0:
        vx_cmd = max(vx_cmd, min_vx)
    return vx_cmd if distance_error > 0.0 else -vx_cmd


# ============================================================================
# MAVLink send helpers
# ============================================================================


def send_set_position_target_local_ned(master, vx, vy, vz, yaw_rate_rad):
    """Body-frame velocity setpoint (msg 84). Velocity + yaw_rate only."""
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        MAV_FRAME_BODY_NED,
        TYPE_MASK_VELOCITY_YAW_RATE,
        0.0, 0.0, 0.0,
        float(vx), float(vy), float(vz),
        0.0, 0.0, 0.0,
        0.0,
        float(yaw_rate_rad),
    )


def send_mount_control(master, pitch_deg: float, yaw_deg: float) -> None:
    """MAV_CMD_DO_MOUNT_CONTROL (205). Absolute gimbal angles via MAVLink targeting."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_DO_MOUNT_CONTROL,
        0,
        float(pitch_deg),
        0.0,
        float(yaw_deg),
        0.0, 0.0, 0.0,
        float(MAV_MOUNT_MODE_MAVLINK_TARGETING),
    )


def send_relay(master, relay_num: int, state: int) -> None:
    """MAV_CMD_DO_SET_RELAY (181). Safety OFF after a burst and on script exit."""
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
    """MAV_CMD_DO_REPEAT_RELAY (182). Mirrors QGroundControl 'Shoot Gun' action.
    Re-used here as the water-solenoid discharge trigger."""
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
    """MAV_CMD_SET_MESSAGE_INTERVAL (511). Request msg_id at interval microseconds."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        float(int(msg_id)),
        float(int(interval_us)),
        0.0, 0.0, 0.0, 0.0, 0.0,
    )


def request_streams(master, rate_hz: int) -> None:
    """Legacy request_data_stream — belt-and-suspenders alongside SET_MESSAGE_INTERVAL."""
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


# ============================================================================
# MovementLink — drone velocity setpoints
# ============================================================================


class MovementLink:
    """Owns the drone-velocity side of the MAVLink connection.

    State (lock-protected):
      - _vx, _vy, _vz, _yaw_rate_rad   : current velocity setpoint (m/s, rad/s)
      - _last_heartbeat                : (flightmode, base_mode, custom_mode, sys_status, mono_time)
      - _last_distance_cm              : (cm, mono_time)
      - _last_vfr_hud                  : (gs, as_, climb, alt, hdg, mono_time)
      - _last_local_position           : (x, y, z, vx, vy, vz, mono_time)
      - _last_command_ack              : (cmd, result, mono_time)

    Threading:
      - inference_thread calls set_velocity(vx, vy, vz, yaw_rate_dps) each frame
      - drone_tx_thread (run_tx_loop) transmits at send_rate_hz
      - shared_rx_thread (NOT this class's run_rx_loop) calls handle_message(msg)
        for each inbound message — this avoids racing recv_match across threads.

    Mode gating: tx_loop checks self.is_guided() before each send. If not GUIDED,
    skips the send and logs (rate-limited) [MODE LOST]."""

    def __init__(
        self,
        master,
        send_rate_hz: float,
        dry_run: bool,
        require_guided_to_send: bool = True,
        rangefinder_orientation: int = 0,
    ):
        self.master = master
        self.dry_run = dry_run
        self.require_guided_to_send = require_guided_to_send
        self.send_interval = 1.0 / max(1.0, float(send_rate_hz))
        self.rangefinder_orientation = int(rangefinder_orientation)
        self._rejected_orientations = {}

        self._lock = threading.Lock()
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._yaw_rate_rad = 0.0

        self._last_heartbeat = None
        self._last_distance_cm = None
        self._last_vfr_hud = None
        self._last_local_position = None
        self._last_command_ack = None

        self.send_count = 0
        self.blank_send_count = 0
        self.last_send_time = None
        self.last_send_err = None
        self.last_mode_lost_log = 0.0

    # --- setters / getters ---

    def set_velocity(self, vx, vy, vz, yaw_rate_dps):
        with self._lock:
            self._vx = float(vx)
            self._vy = float(vy)
            self._vz = float(vz)
            self._yaw_rate_rad = math.radians(float(yaw_rate_dps))

    def get_velocity(self):
        with self._lock:
            return self._vx, self._vy, self._vz, math.degrees(self._yaw_rate_rad)

    def get_velocity_raw(self):
        with self._lock:
            return self._vx, self._vy, self._vz, self._yaw_rate_rad

    def get_mode(self):
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
            return False
        return mode == "GUIDED"

    def get_distance_cm(self):
        with self._lock:
            if self._last_distance_cm is None:
                return None, None
            cm, ts = self._last_distance_cm
            return cm, time.monotonic() - ts

    def get_vfr_hud(self):
        with self._lock:
            if self._last_vfr_hud is None:
                return None, None, None, None, None, None
            gs, as_, climb, alt, hdg, ts = self._last_vfr_hud
            return gs, as_, climb, alt, hdg, time.monotonic() - ts

    def get_local_position(self):
        with self._lock:
            if self._last_local_position is None:
                return None, None, None, None, None, None, None
            x, y, z, vx, vy, vz, ts = self._last_local_position
            return x, y, z, vx, vy, vz, time.monotonic() - ts

    # --- transmit ---

    def _transmit(self, vx, vy, vz, yaw_rate_rad) -> bool:
        if self.dry_run or self.master is None:
            self.blank_send_count += 1
            self.last_send_time = time.monotonic()
            return False
        if self.require_guided_to_send and not self.is_guided():
            now = time.monotonic()
            if (now - self.last_mode_lost_log) >= 2.0:
                mode, _ = self.get_mode()
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
        ArduPilot expects >=4 Hz of SET_POSITION_TARGET_LOCAL_NED to maintain
        GUIDED motion; without them it eventually times out the setpoint."""
        last_send = 0.0
        blank_log_interval = 1.0
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
                if (not sent) and self.dry_run:
                    if (now - last_blank_log) >= blank_log_interval:
                        yaw_rate_dps = math.degrees(yaw_rate_rad)
                        print(f"[BLANK SEND] SET_POSITION_TARGET_LOCAL_NED "
                              f"frame=BODY_NED type_mask=0x{TYPE_MASK_VELOCITY_YAW_RATE:03X} "
                              f"vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} "
                              f"yaw_rate={yaw_rate_dps:+.2f}deg/s "
                              f"blank_total={self.blank_send_count}", flush=True)
                        last_blank_log = now
            time.sleep(0.005)

    def send_halt(self) -> bool:
        """Single zero-velocity setpoint to halt the drone immediately."""
        with self._lock:
            self._vx = 0.0
            self._vy = 0.0
            self._vz = 0.0
            self._yaw_rate_rad = 0.0
        if self.master is None or self.dry_run:
            print("[HALT] BLANK or no MAVLink — would send vx=vy=vz=0, yaw_rate=0", flush=True)
            return False
        try:
            send_set_position_target_local_ned(self.master, 0.0, 0.0, 0.0, 0.0)
            print("[HALT] sent zero-velocity SET_POSITION_TARGET_LOCAL_NED", flush=True)
            return True
        except Exception as e:
            print(f"[HALT] WARN: zero-velocity send failed: {e}", flush=True)
            return False

    # --- inbound message handler (called by shared_rx_loop, NOT run_rx_loop) ---

    def handle_message(self, msg, on_handback_mode_ack=None) -> None:
        """Capture telemetry by msg type. Called once per message by the shared
        rx loop. on_handback_mode_ack is an optional callback invoked when a
        COMMAND_ACK for DO_SET_MODE (cmd 176) arrives — the mission layer uses
        this to confirm PHASE_HANDBACK's mode switch."""
        t = msg.get_type()
        now = time.monotonic()
        if t == "HEARTBEAT":
            try:
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
                orientation = int(getattr(msg, "orientation", 0))
                if (
                    self.rangefinder_orientation != -1
                    and orientation != self.rangefinder_orientation
                ):
                    last = self._rejected_orientations.get(orientation, 0.0)
                    if now - last > 5.0:
                        self._rejected_orientations[orientation] = now
                        print(f"[RANGEFINDER] Ignoring DISTANCE_SENSOR with "
                              f"orientation={orientation} "
                              f"(want={self.rangefinder_orientation}). "
                              f"Override with --rangefinder-orientation.",
                              flush=True)
                    return
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
                if cmd == MAV_CMD_SET_MESSAGE_INTERVAL:
                    print(f"[ACK] SET_MESSAGE_INTERVAL result={result}", flush=True)
                elif cmd == MAV_CMD_DO_SET_MODE:
                    print(f"[ACK] DO_SET_MODE result={result}", flush=True)
                    if on_handback_mode_ack is not None:
                        try:
                            on_handback_mode_ack(result, now)
                        except Exception as e:
                            print(f"[ACK] handback callback failed: {e}", flush=True)
            except Exception:
                pass


# ============================================================================
# GimbalLink — slew-rate-limited gimbal aiming
# ============================================================================


class GimbalLink:
    """Owns the gimbal side of the MAVLink connection (cmd 205, cmd 181/182).

    Two setpoints, lock-protected:
      - wished_pitch / wished_yaw   : controller's desired angles (inference thread)
      - current_pitch / current_yaw : what we've actually transmitted (TX thread)

    Each TX tick, current_* is stepped toward wished_* by at most step_per_tick
    degrees. This rate-limits the transmitted setpoint to the gimbal's physical
    slew capability. The inference thread can set wished_* freely without
    overshoot.

    The rx side (handle_message) captures RELAY_STATUS, DO_REPEAT_RELAY/DO_SET_RELAY
    ACKs, and STATUSTEXT for fire-state-machine confirmation. Called by the shared
    rx loop, not by this class's run_rx_loop.
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

        self._relay_on_mask = 0
        self._relay_present_mask = 0
        self._relay_status_time = None

        self._last_repeat_relay_ack_result = None
        self._last_repeat_relay_ack_time = None

    def get_relay_status(self):
        with self._lock:
            if self._relay_status_time is None:
                return self._relay_on_mask, self._relay_present_mask, None
            return (self._relay_on_mask, self._relay_present_mask,
                    time.monotonic() - self._relay_status_time)

    def get_last_repeat_relay_ack(self):
        with self._lock:
            return self._last_repeat_relay_ack_result, self._last_repeat_relay_ack_time

    def set_wished(self, pitch: float, yaw: float) -> None:
        with self._lock:
            self._wished_pitch = float(pitch)
            self._wished_yaw = float(yaw)

    def get_current(self):
        with self._lock:
            return self._pitch, self._yaw

    def get_wished(self):
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
        """Slew-rate-limited transmit. Each tick, ramp current toward wished by
        at most step_per_tick degrees, then transmit (if moved) or heartbeat-send
        (if idle past heartbeat_send_interval)."""
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

    def handle_message(self, msg) -> None:
        """Capture gimbal-relevant telemetry. Called by shared_rx_loop."""
        t = msg.get_type()
        if t == "COMMAND_ACK":
            try:
                cmd = int(msg.command)
                if cmd == MAV_CMD_DO_MOUNT_CONTROL:
                    print(f"[ACK] cmd={cmd} result={msg.result}", flush=True)
                elif cmd == MAV_CMD_DO_SET_RELAY or cmd == MAV_CMD_DO_REPEAT_RELAY:
                    print(f"[ACK-RELAY] cmd={cmd} result={msg.result} "
                          f"(0=ACCEPTED, 4=FAILED, 2=DENIED, 5=UNSUPPORTED)",
                          flush=True)
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


# ============================================================================
# Module-level helpers shared between phases
# ============================================================================


def read_current_gimbal_position(master, timeout: float):
    """Drain inbox looking for MOUNT_STATUS or GIMBAL_DEVICE_ATTITUDE_STATUS.
    Returns (pitch_deg, yaw_deg) or None. Used by --start-from-current-gimbal."""
    if master is None:
        return None

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
                yaw_rad = math.atan2(2.0 * (w * z + x * y),
                                     1.0 - 2.0 * (y * y + z * z))
                return math.degrees(pitch_rad), math.degrees(yaw_rad)
            except Exception:
                continue
    return None


def center_gimbal_at_start(master, duration: float, rate_hz: float, dry_run: bool) -> None:
    """Send (pitch=0, yaw=0) at rate_hz for duration seconds so the gimbal
    reaches a known reference before tracking begins."""
    interval = 1.0 / max(0.1, float(rate_hz))
    sends = max(1, int(round(duration / interval)))
    print(f"[CENTER-GIMBAL] target (pitch=0.00, yaw=0.00) sends={sends} "
          f"rate={rate_hz}Hz duration={duration:.2f}s dry_run={dry_run}", flush=True)
    for i in range(sends):
        if dry_run or master is None:
            print(f"[CENTER-GIMBAL] DRY {i+1}/{sends} CMD_LONG cmd=205 "
                  f"param1=0.00 param3=0.00 param7={MAV_MOUNT_MODE_MAVLINK_TARGETING}",
                  flush=True)
        else:
            try:
                send_mount_control(master, 0.0, 0.0)
                print(f"[CENTER-GIMBAL] TX  {i+1}/{sends} pitch=0.00 yaw=0.00",
                      flush=True)
            except Exception as e:
                print(f"[CENTER-GIMBAL] send_mount_control failed: {e}", flush=True)
        time.sleep(interval)
    print("[CENTER-GIMBAL] done", flush=True)


def open_mavlink_and_check_guided(
    connection_str: str,
    heartbeat_timeout: float,
    stream_rate: int,
    require_guided: bool,
):
    """Connect, wait heartbeat, verify GUIDED, subscribe drone-side + relay
    telemetry. Returns (master, flightmode_str). Aborts with EXIT_GUIDED_CHECK_FAILED
    if require_guided=True and mode != GUIDED."""
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[MAVLINK] pymavlink not installed. Re-run with --no-mavlink.",
              flush=True)
        return None, None

    print(f"[MAVLINK] Connecting: {connection_str}", flush=True)
    try:
        master = mavutil.mavlink_connection(connection_str)
        master.wait_heartbeat(timeout=heartbeat_timeout)
        print(f"[MAVLINK] heartbeat sysid={master.target_system} "
              f"compid={master.target_component}", flush=True)
    except Exception as e:
        print(f"[MAVLINK] WARNING: connection/heartbeat failed: {e}", flush=True)
        return None, None

    mode = getattr(master, "flightmode", None)
    print(f"[MAVLINK] autopilot flightmode={mode}", flush=True)

    if require_guided and mode != "GUIDED":
        print(
            f"\n[GUIDED CHECK FAILED] Autopilot is in flightmode={mode}.\n"
            f"Switch to GUIDED via QGroundControl or transmitter, then re-run.\n"
            f"This script does NOT auto-switch INTO GUIDED — operator action only.\n"
            f"(It WILL switch OUT of GUIDED at the end via PHASE_HANDBACK.)\n",
            flush=True,
        )
        try:
            master.close()
        except Exception:
            pass
        sys.exit(EXIT_GUIDED_CHECK_FAILED)

    request_streams(master, stream_rate)

    # Drone-side telemetry subscriptions
    for msg_id, label in (
        (MAVLINK_MSG_ID_HEARTBEAT, "HEARTBEAT @ 4 Hz (for mode monitoring)"),
        (MAVLINK_MSG_ID_DISTANCE_SENSOR, "DISTANCE_SENSOR @ 5 Hz (range-to-target)"),
        (MAVLINK_MSG_ID_LOCAL_POSITION_NED, "LOCAL_POSITION_NED @ 5 Hz"),
        (MAVLINK_MSG_ID_VFR_HUD, "VFR_HUD @ 5 Hz (groundspeed/climb feedback)"),
    ):
        interval_us = 250000 if msg_id == MAVLINK_MSG_ID_HEARTBEAT else 200000
        try:
            set_message_interval(master, msg_id, interval_us)
            print(f"[MAVLINK] Requested {label}", flush=True)
        except Exception as e:
            print(f"[MAVLINK] WARNING: SET_MESSAGE_INTERVAL for msg {msg_id} failed: {e}",
                  flush=True)

    # Gimbal-side telemetry subscriptions (RELAY_STATUS for visibility)
    try:
        set_message_interval(master, MAVLINK_MSG_ID_RELAY_STATUS, 200000)
        print(f"[MAVLINK] Requested RELAY_STATUS (msg {MAVLINK_MSG_ID_RELAY_STATUS}) "
              f"@ 5 Hz", flush=True)
    except Exception as e:
        print(f"[MAVLINK] WARNING: SET_MESSAGE_INTERVAL for RELAY_STATUS failed: {e}",
              flush=True)

    return master, mode


def shared_rx_loop(master, drone_link, gimbal_link, on_handback_mode_ack,
                   stop_event: threading.Event) -> None:
    """SINGLE rx loop for both Link classes. pymavlink's recv_match is not safe
    to race across threads on the same socket, so we drain it in one place and
    dispatch each message to the correct Link (or both, for COMMAND_ACK)."""
    if master is None:
        return
    while not stop_event.is_set():
        try:
            msg = master.recv_match(blocking=True, timeout=0.5)
        except Exception as e:
            print(f"[MAVLINK] recv_match failed: {e}", flush=True)
            time.sleep(0.2)
            continue
        if msg is None or msg.get_type() == "BAD_DATA":
            continue
        t = msg.get_type()

        # Drone-side messages
        if t in ("HEARTBEAT", "DISTANCE_SENSOR", "VFR_HUD", "LOCAL_POSITION_NED"):
            drone_link.handle_message(msg, on_handback_mode_ack=on_handback_mode_ack)
        # Gimbal-side messages
        elif t == "RELAY_STATUS":
            gimbal_link.handle_message(msg)
        # Shared: COMMAND_ACK — branch by cmd ID
        elif t == "COMMAND_ACK":
            try:
                cmd = int(msg.command)
            except Exception:
                cmd = -1
            if cmd in (MAV_CMD_DO_SET_RELAY, MAV_CMD_DO_REPEAT_RELAY,
                       MAV_CMD_DO_MOUNT_CONTROL):
                gimbal_link.handle_message(msg)
            else:
                # Cmd 176 (DO_SET_MODE), 511 (SET_MESSAGE_INTERVAL), or others
                drone_link.handle_message(msg, on_handback_mode_ack=on_handback_mode_ack)
        elif t == "STATUSTEXT":
            text = (getattr(msg, "text", "") or "").strip()
            if text:
                print(f"[STATUSTEXT sev={getattr(msg, 'severity', '?')}] {text}",
                      flush=True)
        # Other messages (MOUNT_STATUS, GIMBAL_DEVICE_ATTITUDE_STATUS, etc.) are
        # silently consumed — they're only read at startup via
        # read_current_gimbal_position(), not during the run.


# ============================================================================
# main()
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        prog="tflive_final_auto",
        description="Big City RPAS Task 2 — combined autonomous drone-positioning "
                    "+ gimbal-aim + water-discharge + image-capture mission script.",
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
                        help="Enable inference processing (REQUIRED for engagement)")
    parser.add_argument("--overlay", "-o", action="store_true",
                        help="Draw detection labels/boxes + mission OSD on output video")
    parser.add_argument("--no-output", action="store_true",
                        help="Disable video output stream (capture + inference only)")
    parser.add_argument("--video", type=str, default=None,
                        help="Override RTSP source with a local video index or path "
                             "(used by --no-mavlink laptop dry-runs).")

    # --- Detection post-filtering ---
    parser.add_argument("--min-conf", type=float, default=0.05,
                        help="Post-filter minimum confidence")
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01)
    parser.add_argument("--min-track-confidence", type=float, default=0.35,
                        help="Secondary confidence gate for drone state machine")

    # --- Crop passes ---
    parser.add_argument("--center-crop-pass", action="store_true")
    parser.add_argument("--center-crop-ratio", type=float, default=0.5)
    parser.add_argument("--crop-center-x", type=float, default=0.5)
    parser.add_argument("--crop-center-y", type=float, default=0.5)
    parser.add_argument("--second-crop-pass", action="store_true")
    parser.add_argument("--second-crop-ratio", type=float, default=0.45)
    parser.add_argument("--second-crop-center-x", type=float, default=0.55)
    parser.add_argument("--second-crop-center-y", type=float, default=0.35)

    # --- Preprocessing ---
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--luminance", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--sharpen", type=float, default=0.0,
                        help="Unsharp-mask amount. Production recipe: --sharpen 0.4.")
    parser.add_argument("--sharpen-sigma", type=float, default=1.0)

    # --- MAVLink connection ---
    parser.add_argument("--mavlink", type=str, default="tcp:10.42.0.1:5760")
    parser.add_argument("--no-mavlink", action="store_true",
                        help="Skip MAVLink entirely; pure dry-run with simulated LiDAR.")
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0)
    parser.add_argument("--stream-rate", type=int, default=10)

    # --- Drone control / live-fly gate ---
    parser.add_argument("--live-fly", action="store_true",
                        help="DANGEROUS. Send real SET_POSITION_TARGET_LOCAL_NED. "
                             "Default = BLANK (read telemetry, log what we would send).")
    parser.add_argument("--no-guided-check", action="store_true",
                        help="Skip startup GUIDED-mode check (ground bench only).")
    parser.add_argument("--tx-rate", type=float, default=10.0,
                        help="Drone setpoint send rate (Hz). Must be >= 4 Hz.")

    # Drone-namespace gains (renamed to avoid collision with gimbal-namespace versions)
    parser.add_argument("--drone-deadband", type=float, default=0.08,
                        help="Normalized deadband for drone yaw alignment (default 0.08).")
    parser.add_argument("--camera-hfov-deg", type=float, default=78.0)

    # --- Anti-jitter: bbox EMA + yaw-aligned hysteresis + NO_TARGET dropout grace ---
    # Same fix as drone_auto 2026-05-18 — see those notes. Applies to the drone
    # state machine in PHASE_DRONE_POSITIONING / PHASE_HANDOFF_WAIT only.
    parser.add_argument("--bbox-ema-alpha", type=float, default=0.5,
                        help="EMA weight on bbox center per new frame. 1.0=raw "
                             "(no smoothing), lower=more smoothing. Default 0.5.")
    parser.add_argument("--bbox-ema-reset-frames", type=int, default=3,
                        help="Consecutive NO_TARGET frames before EMA resets. "
                             "Default 3.")
    parser.add_argument("--yaw-hysteresis-ratio", type=float, default=2.0,
                        help="Exit-deadband / entry-deadband ratio. Entry="
                             "drone_deadband, Exit=drone_deadband*ratio. Default 2.0.")
    parser.add_argument("--yaw-align-dropout-frames", type=int, default=3,
                        help="Frames with |err_x|>exit-deadband before latch releases. "
                             "Default 3.")
    parser.add_argument("--no-target-dropout-frames", type=int, default=3,
                        help="NO_TARGET frames before hold_confirm_count / EMA / latch "
                             "reset. Default 3.")
    parser.add_argument("--drone-yaw-gain", type=float, default=35.0,
                        help="Drone yaw-rate gain in deg/s at full err_x (default 35).")
    parser.add_argument("--max-yaw-rate", type=float, default=25.0)
    parser.add_argument("--forward-gain", type=float, default=0.80)
    parser.add_argument("--max-vx", type=float, default=0.55)
    parser.add_argument("--min-distance-correct-vx", type=float, default=0.08)
    parser.add_argument("--target-distance-cm", type=float, default=200.0)
    parser.add_argument("--distance-tolerance-cm", type=float, default=15.0)
    parser.add_argument("--distance-center-band-cm", type=float, default=2.0)
    parser.add_argument("--hold-forward-gain", type=float, default=0.35)
    parser.add_argument("--hold-min-distance-correct-vx", type=float, default=0.03)
    parser.add_argument("--lock-after-approach", action="store_true", default=True)
    parser.add_argument("--no-lock-after-approach", action="store_false",
                        dest="lock_after_approach")
    parser.add_argument("--lock-confirm-frames", type=int, default=8)
    parser.add_argument("--lock-yaw-gain", type=float, default=45.0)
    parser.add_argument("--lock-forward-gain", type=float, default=1.00)
    parser.add_argument("--lock-deadband-scale", type=float, default=0.50)
    parser.add_argument("--altitude-target-y-ratio", type=float, default=0.75)
    parser.add_argument("--altitude-deadband", type=float, default=0.04)
    parser.add_argument("--altitude-gain", type=float, default=0.55)
    parser.add_argument("--max-vz", type=float, default=0.35)
    parser.add_argument("--altitude-lock-confirm-frames", type=int, default=8)

    # --- Distance source ---
    parser.add_argument("--simulate-distance", action="store_true",
                        help="Use simulated LiDAR. Auto-enabled by --no-mavlink.")
    parser.add_argument("--rangefinder-orientation", type=int, default=0,
                        help="MAV_SENSOR_ORIENTATION filter for DISTANCE_SENSOR "
                             "(0=forward, 25=down, -1=any).")
    parser.add_argument("--distance-sensor-timeout", type=float, default=5.0)

    # --- Simulated-disturbance args (--simulate-distance only) ---
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

    # --- Gimbal control (namespace flags renamed where collision existed) ---
    parser.add_argument("--gimbal-deadband", type=float, default=0.08,
                        help="Normalized image-error deadband for gimbal CENTERED label.")
    parser.add_argument("--gimbal-yaw-gain", type=float, default=12.0,
                        help="Gimbal yaw gain — wished_yaw = current_yaw + gain*err_x.")
    parser.add_argument("--pitch-gain", type=float, default=10.0,
                        help="Gimbal pitch gain — wished_pitch = current_pitch - gain*err_y.")
    parser.add_argument("--max-slew-rate-yaw", type=float, default=2.0,
                        help="Max gimbal yaw slew rate (deg/sec). Default 2.")
    parser.add_argument("--max-slew-rate-pitch", type=float, default=1.5,
                        help="Max gimbal pitch slew rate (deg/sec). Default 1.5.")
    parser.add_argument("--send-rate", type=float, default=20.0,
                        help="Gimbal command send rate when moving (Hz). Default 20.")
    parser.add_argument("--heartbeat-send-rate", type=float, default=2.0,
                        help="Gimbal idle resend rate (Hz). Default 2.")

    # Gimbal startup mode — mutually exclusive
    parser.add_argument("--center-gimbal-at-start", action="store_true", default=True,
                        help="At startup, send (pitch=0, yaw=0) for --center-duration "
                             "seconds. Default behavior.")
    parser.add_argument("--no-center-gimbal-at-start", action="store_false",
                        dest="center_gimbal_at_start")
    parser.add_argument("--start-from-current-gimbal", action="store_true",
                        help="At startup, read MOUNT_STATUS / GIMBAL_DEVICE_ATTITUDE_STATUS "
                             "once. Mutex with --center-gimbal-at-start.")
    parser.add_argument("--read-current-timeout", type=float, default=2.0)
    parser.add_argument("--center-duration", type=float, default=2.0)
    parser.add_argument("--center-rate", type=float, default=5.0)
    parser.add_argument("--initial-pitch", type=float, default=0.0)
    parser.add_argument("--initial-yaw", type=float, default=0.0)

    # --- Fire control (water discharge in Task 2 context) ---
    parser.add_argument("--no-fire", action="store_true",
                        help="Disable all discharge logic (tracking-only run).")
    parser.add_argument("--live-fire", action="store_true",
                        help="DANGEROUS. Send real DO_REPEAT_RELAY to trigger the "
                             "water solenoid. Default = BLANK (simulated phases).")
    parser.add_argument("--fire-relay", type=int, default=1,
                        help="Relay instance (default 1, matches QGC 'Shoot Gun').")
    parser.add_argument("--fire-period", type=float, default=5.0,
                        help="Seconds the autopilot holds the relay ON per burst. "
                             "Default 5.0 (cycle = 10s = 5s ON + 5s OFF).")
    parser.add_argument("--fire-cooldown", type=float, default=0.5,
                        help="Additional idle after full cycle before re-arm. "
                             "Not used in this script — mission exits after one burst.")

    # --- Combined-mission flags ---
    parser.add_argument("--handoff-confirm-frames", type=int, default=15,
                        help="Consecutive FINAL_HOLD frames required to confirm "
                             "drone-positioning -> gimbal-tracking handoff. "
                             "Default 15 (~1.5s at 10 Hz).")
    parser.add_argument("--handback-mode", type=str, default="LOITER",
                        choices=["LOITER", "RTL", "ALT_HOLD", "LAND"],
                        help="Autopilot flight mode requested at end-of-mission via "
                             "MAV_CMD_DO_SET_MODE. LOITER = pilot resumes manual "
                             "control. LAND = autonomous-landing 5-pt criterion.")
    parser.add_argument("--handback-mode-timeout", type=float, default=2.0,
                        help="Seconds to wait for DO_SET_MODE ACK before exiting "
                             "anyway. Default 2.0.")
    parser.add_argument("--min-start-distance-cm", type=float, default=200.0,
                        help="Task 2 compliance gate: first DISTANCE_SENSOR reading "
                             "must be >= this value (Task 2 §5.2.4 requires the "
                             "autonomous approach to start from >2 m). Default 200 cm. "
                             "Set to 0 to disable the gate for laptop testing.")

    # --- Task 2 photo capture ---
    parser.add_argument("--team-name", type=str, default="unknown",
                        help="Team name in Task 2 photo filename "
                             "Task_2_<team_name>_target_<#>_<ts>.jpg. "
                             "Set on every live run!")
    parser.add_argument("--target-number", type=int, default=1,
                        help="Target index in Task 2 photo filename. Increment "
                             "between successive engagements within a flight window.")
    parser.add_argument("--photo-output-dir", type=str, default="./extinguish_photos",
                        help="Directory for saved JPEGs. Created if absent.")
    parser.add_argument("--no-photo-capture", action="store_true",
                        help="Skip PHASE_VERIFY entirely (NOT Task-2-compliant; "
                             "use only for testing without team-name set).")
    parser.add_argument("--capture-frame-count", type=int, default=5,
                        help="Frames evaluated during PHASE_VERIFY. Default 5.")
    parser.add_argument("--capture-frame-interval", type=float, default=0.4,
                        help="Spacing between PHASE_VERIFY captures (sec). Default 0.4 "
                             "(0.4 * 5 = 2.0 s total verify window).")

    args = parser.parse_args()

    # --- Validation ---
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
    if args.drone_deadband < 0.0 or args.drone_deadband >= 1.0:
        parser.error("--drone-deadband must be >= 0.0 and < 1.0")
    if not (0.0 < args.bbox_ema_alpha <= 1.0):
        parser.error("--bbox-ema-alpha must be in (0.0, 1.0]")
    if args.bbox_ema_reset_frames < 1:
        parser.error("--bbox-ema-reset-frames must be >= 1")
    if args.yaw_hysteresis_ratio < 1.0:
        parser.error("--yaw-hysteresis-ratio must be >= 1.0")
    if args.yaw_align_dropout_frames < 1:
        parser.error("--yaw-align-dropout-frames must be >= 1")
    if args.no_target_dropout_frames < 1:
        parser.error("--no-target-dropout-frames must be >= 1")
    if args.gimbal_deadband < 0.0 or args.gimbal_deadband >= 1.0:
        parser.error("--gimbal-deadband must be >= 0.0 and < 1.0")
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
        parser.error("--fire-relay must be in [0,15]")
    if args.handoff_confirm_frames < 1:
        parser.error("--handoff-confirm-frames must be >= 1")
    if args.handback_mode_timeout <= 0.0:
        parser.error("--handback-mode-timeout must be > 0")
    if args.capture_frame_count < 1:
        parser.error("--capture-frame-count must be >= 1")
    if args.capture_frame_interval <= 0.0:
        parser.error("--capture-frame-interval must be > 0")
    if args.live_fly and args.no_mavlink:
        print("[ERROR] --live-fly and --no-mavlink are mutually exclusive.", flush=True)
        sys.exit(EXIT_ARG_CONFLICT)
    if args.live_fly and args.simulate_distance:
        print("[ERROR] --live-fly and --simulate-distance are mutually exclusive.",
              flush=True)
        sys.exit(EXIT_ARG_CONFLICT)
    if args.start_from_current_gimbal and args.center_gimbal_at_start:
        # User explicitly asked for from-current; suppress the default center
        args.center_gimbal_at_start = False
    if args.no_mavlink:
        args.simulate_distance = True

    for n, v in (("--luminance", args.luminance), ("--contrast", args.contrast),
                 ("--saturation", args.saturation), ("--sharpen", args.sharpen),
                 ("--sharpen-sigma", args.sharpen_sigma)):
        if v < 0.0:
            parser.error(f"{n} must be >= 0")

    # Task 2 compliance warnings (don't reject — operator may know what they're doing)
    if args.min_start_distance_cm < 200.0 and args.min_start_distance_cm > 0.0:
        print(f"[WARN] --min-start-distance-cm={args.min_start_distance_cm:.0f} is "
              f"below the Task 2 §5.2.4 minimum of 200 cm. Autonomous-extinguishing "
              f"credit requires >2 m approach start.", flush=True)
    if args.live_fire and args.team_name == "unknown" and not args.no_photo_capture:
        print("[WARN] --live-fire set but --team-name='unknown'. The Task 2 photo "
              "will be saved as 'Task_2_unknown_target_<N>_<ts>.jpg'. Set --team-name "
              "to your team identifier before live invocation.", flush=True)

    # --- Preproc banner ---
    preproc_parts = []
    if args.grayscale: preproc_parts.append("grayscale")
    if args.luminance != 1.0: preproc_parts.append(f"luminance={args.luminance:.2f}")
    if args.contrast != 1.0: preproc_parts.append(f"contrast={args.contrast:.2f}")
    if args.saturation != 1.0 and not args.grayscale:
        preproc_parts.append(f"saturation={args.saturation:.2f}")
    if args.sharpen > 0.0:
        preproc_parts.append(f"sharpen={args.sharpen:.2f}(sigma={args.sharpen_sigma:.1f})")
    if args.clahe:
        preproc_parts.append(f"clahe(clip={args.clahe_clip_limit:.1f},grid={args.clahe_grid})")
    if preproc_parts:
        print(f"[PREPROC] {' | '.join(preproc_parts)}", flush=True)
    else:
        print("[PREPROC] (none — defaults)", flush=True)

    # --- Camera ---
    if args.video is not None:
        # Local source (laptop dry-run convenience)
        try:
            video_src = int(args.video)
        except ValueError:
            video_src = args.video
        cap = cv2.VideoCapture(video_src)
    else:
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
        print("[ERROR] This script requires --process for autonomous engagement.",
              flush=True)
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

    # --- MAVLink + GUIDED check ---
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

    # --- Wait for first DISTANCE_SENSOR (real mode only) ---
    if not args.simulate_distance:
        want_orient = args.rangefinder_orientation
        orient_desc = ("any orientation" if want_orient == -1
                       else f"orientation={want_orient}")
        print(f"[INIT] Waiting up to {args.distance_sensor_timeout:.1f}s for first "
              f"DISTANCE_SENSOR ({orient_desc})...", flush=True)
        end_t = time.monotonic() + args.distance_sensor_timeout
        got = False
        seen_orientations = set()
        while time.monotonic() < end_t:
            try:
                msg = master.recv_match(blocking=True, timeout=0.3)
            except Exception:
                msg = None
            if msg is not None and msg.get_type() == "DISTANCE_SENSOR":
                msg_orient = int(getattr(msg, "orientation", 0))
                seen_orientations.add(msg_orient)
                if want_orient != -1 and msg_orient != want_orient:
                    continue
                cm = float(getattr(msg, "current_distance", 0))
                print(f"[INIT] DISTANCE_SENSOR seen: {cm:.1f} cm "
                      f"(orientation={msg_orient})", flush=True)
                got = True
                break
        if not got:
            seen_str = (", ".join(str(o) for o in sorted(seen_orientations))
                        if seen_orientations else "none")
            print(f"[ERROR] No DISTANCE_SENSOR with {orient_desc} within "
                  f"{args.distance_sensor_timeout:.1f}s. Seen: {seen_str}.",
                  flush=True)
            try:
                master.close()
            except Exception:
                pass
            sys.exit(EXIT_DISTANCE_SENSOR_TIMEOUT)

    # --- Gimbal initial pose ---
    initial_pitch = clamp(args.initial_pitch, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
    initial_yaw = clamp(args.initial_yaw, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
    if args.start_from_current_gimbal:
        if master is None:
            print("[INIT] --start-from-current-gimbal requested but no MAVLink "
                  "connection; falling back to (initial_pitch, initial_yaw).",
                  flush=True)
        else:
            print(f"[INIT] Reading current gimbal orientation (timeout "
                  f"{args.read_current_timeout:.1f}s)...", flush=True)
            pos = read_current_gimbal_position(master, args.read_current_timeout)
            if pos is None:
                print("[INIT] ERROR: no MOUNT_STATUS or GIMBAL_DEVICE_ATTITUDE_STATUS. "
                      "Re-run without --start-from-current-gimbal to center first instead.",
                      flush=True)
                try:
                    master.close()
                except Exception:
                    pass
                sys.exit(EXIT_GIMBAL_INIT_FAIL)
            initial_pitch = clamp(pos[0], GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
            initial_yaw = clamp(pos[1], GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
            print(f"[INIT] Read current gimbal: pitch={initial_pitch:+.2f} "
                  f"yaw={initial_yaw:+.2f}", flush=True)
    elif args.center_gimbal_at_start:
        if master is not None and not args.no_mavlink:
            print(f"[INIT] Centering gimbal first (pitch=0, yaw=0) for "
                  f"{args.center_duration:.2f}s @ {args.center_rate}Hz...", flush=True)
            center_gimbal_at_start(master, args.center_duration, args.center_rate,
                                   dry_run=args.no_mavlink or master is None)
        initial_pitch = 0.0
        initial_yaw = 0.0
        print(f"[INIT] Initial gimbal reference: pitch={initial_pitch:+.2f} "
              f"yaw={initial_yaw:+.2f}", flush=True)
    else:
        print(f"[INIT] Using --initial-pitch={initial_pitch:+.2f} "
              f"--initial-yaw={initial_yaw:+.2f}", flush=True)

    # --- Link construction ---
    drone_link = MovementLink(
        master=master,
        send_rate_hz=args.tx_rate,
        dry_run=(args.no_mavlink or not args.live_fly),
        require_guided_to_send=(not args.no_guided_check) and (not args.no_mavlink),
        rangefinder_orientation=args.rangefinder_orientation,
    )
    gimbal_link = GimbalLink(
        master=master,
        send_rate_hz=args.send_rate,
        heartbeat_send_rate_hz=args.heartbeat_send_rate,
        max_slew_rate_yaw_dps=args.max_slew_rate_yaw,
        max_slew_rate_pitch_dps=args.max_slew_rate_pitch,
        initial_pitch=initial_pitch,
        initial_yaw=initial_yaw,
        dry_run=(args.no_mavlink or master is None),
    )

    # --- Startup banner ---
    print("=" * 78, flush=True)
    print("[MISSION] Big City RPAS Task 2 — combined autonomous engagement", flush=True)
    print(f"[MISSION] team={args.team_name!r}  target_number={args.target_number}  "
          f"handback_mode={args.handback_mode}", flush=True)
    print(f"[MISSION] drone tx={'LIVE-FLY' if args.live_fly else 'BLANK'}  "
          f"discharge={'LIVE' if args.live_fire else ('DISABLED' if args.no_fire else 'BLANK')}",
          flush=True)
    print(f"[MISSION] photo_capture={'OFF' if args.no_photo_capture else 'ON'}  "
          f"min_start_distance_cm={args.min_start_distance_cm:.0f}", flush=True)
    print("[MISSION] Phase ordering: DRONE_POSITIONING -> HANDOFF_WAIT "
          "-> GIMBAL_TRACKING -> FIRING -> VERIFY -> HANDBACK -> DONE", flush=True)
    print("[INFO] Axis assumptions:", flush=True)
    print("  image: +x=right, +y=down, center=(frame_w/2, frame_h/2)", flush=True)
    print("  drone yaw: target right -> +yaw_rate (clockwise)", flush=True)
    print("  drone vx: +forward closes distance once yaw-aligned", flush=True)
    print("  drone vz: +body-down (descend), -body-up (ascend)", flush=True)
    print("  gimbal yaw: +right/clockwise; gimbal pitch: +up", flush=True)
    if args.live_fly:
        print("[WARN] *** LIVE-FLY ENABLED *** SET_POSITION_TARGET_LOCAL_NED will be "
              "sent at the drone tx rate. Observer must be on RC override.", flush=True)
    if args.live_fire:
        print("[WARN] *** LIVE-FIRE ENABLED *** real DO_REPEAT_RELAY will trigger "
              "the water solenoid in PHASE_FIRING.", flush=True)
    print("=" * 78, flush=True)

    # --- Shared state ---
    latest_frame = [None]
    latest_detections = [[]]
    frame_lock = threading.Lock()
    det_lock = threading.Lock()
    stop_event = threading.Event()
    frame_event = threading.Event()
    frame_seq = [0]

    overlay_state = {
        "state": STATE_NO_TARGET,
        "move_label": "IDLE",
        "displacement_label": "CENTERED",
        "err_x": 0.0, "err_y": 0.0,
        "vx": 0.0, "vz": 0.0, "yaw_rate": 0.0,
        "lidar_cm": 0.0, "lidar_source": "?",
        "dist_status": "?", "dist_error_cm": 0.0,
        "lock_status": "OFF", "final_hold_status": "OFF",
        "hold_confirm_count": 0, "altitude_confirm_count": 0, "lock_age": 0,
        "disturb_vx": 0.0, "disturb_vz": 0.0,
        "yaw_to_center_deg": 0.0,
        "yaw_vec_dx_px": 0.0, "yaw_vec_mag_px": 0.0,
        "alt_vec_dy_px": 0.0, "alt_vec_mag_px": 0.0,
        "body_vec_fwd": 1.0, "body_vec_right": 0.0,
        "target_y_px": 0.0, "alt_error_norm": 0.0,
        "disp_dx_px": 0.0, "disp_dy_px": 0.0, "disp_mag_px": 0.0,
        "tcx": 0, "tcy": 0,
        "frame_idx": 0, "raw_detection_count": 0,
        "actual_vx": None, "actual_vz": None,
        # Gimbal-side
        "gimbal_pitch": initial_pitch, "gimbal_yaw": initial_yaw,
        "gimbal_wished_pitch": initial_pitch, "gimbal_wished_yaw": initial_yaw,
        # Fire-side
        "fire_phase": "IDLE", "fire_elapsed": 0.0,
        # Mission
        "mission_phase": PHASE_DRONE_POSITIONING,
        "final_hold_streak": 0,
        "verify_frames_captured": 0,
        "verify_photo_path": None,
    }
    overlay_lock = threading.Lock()
    mission_lock = threading.Lock()

    sim_state = {
        "lidar_cm": clamp(args.sim_lidar_start_cm,
                          args.sim_lidar_min_cm, args.sim_lidar_max_cm),
        "target_cy": None,
        "last_tick": time.perf_counter(),
    }

    # Mission state — owner: inference thread. Other threads read-only under mission_lock.
    mission_state = {
        "phase": PHASE_DRONE_POSITIONING,
        "phase_entered_at": time.monotonic(),
        "final_hold_streak": 0,
        "handoff_at": None,
        "fire_started_at": None,
        "verify_started_at": None,
        "verify_frames_captured": 0,
        "verify_best_frame": None,    # (np.ndarray, score, mono_time, kind)
        "verify_photo_path": None,
        "verify_done": False,
        "handback_started_at": None,
        "handback_mode_ack": None,    # (result, mono_time)
        "handback_done": False,
        "first_lidar_checked": False,
    }
    run_start_mono = time.monotonic()

    def _transition_phase(new_phase: str, reason: str) -> None:
        with mission_lock:
            old = mission_state["phase"]
            if old == new_phase:
                return
            mission_state["phase"] = new_phase
            mission_state["phase_entered_at"] = time.monotonic()
        elapsed = time.monotonic() - run_start_mono
        print(f"[MISSION t={elapsed:6.2f}s] {old} -> {new_phase} | reason={reason}",
              flush=True)

    def _on_handback_mode_ack(result: int, ts: float) -> None:
        with mission_lock:
            mission_state["handback_mode_ack"] = (int(result), float(ts))

    # --- Fire state machine (verbatim port of gimbal_auto's fire_advance) ---
    fire_burst_duration = float(args.fire_period)
    fire_cycle_time = 2.0 * float(args.fire_period)
    fire_arm_ack_timeout = 1.0
    fire_state = {
        "phase": "IDLE",
        "fire_send_time": 0.0,
    }

    def _enter_fire_phase(new_phase: str, log: str) -> None:
        fire_state["phase"] = new_phase
        print(log, flush=True)

    def fire_advance(centered: bool) -> None:
        """Drive the discharge state machine. Called every inference frame
        while mission_phase in (PHASE_GIMBAL_TRACKING, PHASE_FIRING)."""
        if args.no_fire:
            return
        now = time.monotonic()
        phase = fire_state["phase"]
        mode = "LIVE" if args.live_fire else "BLANK"

        on_mask, _, status_age = gimbal_link.get_relay_status()
        have_fresh_status = status_age is not None and status_age < 2.0
        actual_on = ((on_mask >> args.fire_relay) & 1) if have_fresh_status else None
        relay_tag = (f"RELAY[{args.fire_relay}]="
                     f"{'ON' if actual_on == 1 else 'OFF' if actual_on == 0 else '?'}")

        if phase == "IDLE":
            if centered:
                fire_state["fire_send_time"] = now
                if args.live_fire and master is not None:
                    try:
                        send_repeat_relay(master, args.fire_relay, 1, fire_cycle_time)
                    except Exception as e:
                        print(f"[FIRE ERROR] DO_REPEAT_RELAY failed: {e}", flush=True)
                        return
                _enter_fire_phase(
                    "ARMING",
                    f"[{mode} ARMING] sent DO_REPEAT_RELAY({args.fire_relay},cycles=1,"
                    f"period={fire_cycle_time:.2f}s); awaiting COMMAND_ACK for cmd=182",
                )

        elif phase == "ARMING":
            elapsed = now - fire_state["fire_send_time"]
            if not args.live_fire:
                if elapsed >= 0.1:
                    _enter_fire_phase(
                        "FIRING",
                        f"[{mode} ARMED] BLANK simulated ACK after {elapsed:.2f}s; "
                        f"autopilot would pulse relay ON for ~{fire_burst_duration:.2f}s",
                    )
                return
            ack_result, ack_time = gimbal_link.get_last_repeat_relay_ack()
            ack_is_for_our_send = (
                ack_time is not None and ack_time >= fire_state["fire_send_time"]
            )
            if ack_is_for_our_send:
                if ack_result == 0:
                    _enter_fire_phase(
                        "FIRING",
                        f"[{mode} ARMED] COMMAND_ACK result=0 (ACCEPTED) after "
                        f"{elapsed:.2f}s; autopilot will pulse relay ON for "
                        f"~{fire_burst_duration:.2f}s",
                    )
                else:
                    _enter_fire_phase(
                        "IDLE",
                        f"[{mode} ARMING REJECTED] COMMAND_ACK result={ack_result} "
                        f"(2=DENIED, 4=FAILED, 5=UNSUPPORTED) after {elapsed:.2f}s; "
                        f"NOT firing; back to IDLE",
                    )
            elif elapsed >= fire_arm_ack_timeout:
                _enter_fire_phase(
                    "FIRING",
                    f"[{mode} ARMED WARN] no COMMAND_ACK in {fire_arm_ack_timeout:.2f}s; "
                    f"assuming command got through (ACK may have been lost).",
                )

        elif phase == "FIRING":
            elapsed = now - fire_state["fire_send_time"]
            if elapsed >= fire_burst_duration:
                _enter_fire_phase(
                    "COOLDOWN",
                    f"[{mode} BURST DONE] {fire_burst_duration:.2f}s ON elapsed; "
                    f"{relay_tag}; mission will transition to PHASE_VERIFY",
                )

        elif phase == "COOLDOWN":
            # In final_auto we don't re-arm; PHASE_VERIFY/HANDBACK handles the exit.
            # This branch is a no-op terminal sink.
            pass

    # --- Photo capture (PHASE_VERIFY) ---
    def do_verify_capture(frame, selected, frame_w, frame_h) -> None:
        """Called every inference frame while mission_phase == PHASE_VERIFY.
        Captures --capture-frame-count frames at --capture-frame-interval spacing,
        keeps the best by confidence (Laplacian variance fallback), and saves
        the final JPEG when complete."""
        now = time.monotonic()
        n = mission_state["verify_frames_captured"]
        started = mission_state["verify_started_at"]
        if started is None:
            return
        next_capture_at = started + n * args.capture_frame_interval
        if now < next_capture_at:
            return

        conf = float(selected["confidence"]) if selected is not None else None
        if conf is None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                kind = "sharpness"
            except Exception:
                score = 0.0
                kind = "sharpness_err"
        else:
            score = float(conf)
            kind = "conf"

        prev = mission_state["verify_best_frame"]
        if prev is None or score > prev[1]:
            mission_state["verify_best_frame"] = (frame.copy(), score, now, kind)

        mission_state["verify_frames_captured"] = n + 1
        best_score = mission_state["verify_best_frame"][1]
        print(f"[VERIFY] captured frame {n+1}/{args.capture_frame_count} "
              f"({kind}={score:.3f}); best so far {best_score:.3f}", flush=True)

        if mission_state["verify_frames_captured"] >= args.capture_frame_count:
            try:
                out_dir = pathlib.Path(args.photo_output_dir).expanduser()
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = (f"Task_2_{args.team_name}_target_{args.target_number}_"
                         f"{ts}.jpg")
                path = out_dir / fname
                best = mission_state["verify_best_frame"][0]
                ok = cv2.imwrite(str(path), best,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ok:
                    mission_state["verify_photo_path"] = str(path)
                    print(f"[VERIFY DECLARED] Photo saved at {path}", flush=True)
                    print("[VERIFY] Upload to team Google Drive folder.", flush=True)
                    print("[VERIFY] Visually confirm target turned BLUE before "
                          "declaring to judges (false declaration penalty = "
                          "-P_indoor or -P_outdoor pts).", flush=True)
                else:
                    print(f"[VERIFY ERROR] cv2.imwrite failed for {path}", flush=True)
            except Exception as e:
                print(f"[VERIFY ERROR] capture/save failed: {e}", flush=True)
            mission_state["verify_done"] = True

    # --- Handback (PHASE_HANDBACK) ---
    def do_handback_once() -> None:
        """Idempotent. First call: send DO_SET_MODE, halt drone, defensive relay-OFF.
        Subsequent calls: poll for ACK or timeout."""
        with mission_lock:
            done = mission_state["handback_done"]
            started_at = mission_state["handback_started_at"]
            ack = mission_state["handback_mode_ack"]
        if done:
            return

        if started_at is None:
            ts = time.monotonic()
            with mission_lock:
                mission_state["handback_started_at"] = ts
            print(f"[HANDBACK] initiated; target mode={args.handback_mode}", flush=True)
            if args.live_fly and master is not None:
                try:
                    master.set_mode_apm(args.handback_mode)
                    print(f"[HANDBACK] sent DO_SET_MODE -> {args.handback_mode}",
                          flush=True)
                except Exception as e:
                    print(f"[HANDBACK WARN] set_mode_apm failed: {e}", flush=True)
            else:
                print(f"[HANDBACK BLANK] would request mode={args.handback_mode}",
                      flush=True)
            try:
                drone_link.send_halt()
            except Exception as e:
                print(f"[HANDBACK WARN] send_halt failed: {e}", flush=True)
            if not args.no_fire and master is not None:
                try:
                    send_relay(master, args.fire_relay, 0)
                    print(f"[HANDBACK] sent defensive DO_SET_RELAY({args.fire_relay},0)",
                          flush=True)
                except Exception:
                    pass
            return

        # Subsequent calls — poll for ACK or timeout
        elapsed = time.monotonic() - started_at
        if ack is not None and ack[1] >= started_at:
            with mission_lock:
                mission_state["handback_done"] = True
            print(f"[HANDBACK] DO_SET_MODE ACK result={ack[0]} after {elapsed:.2f}s",
                  flush=True)
            _transition_phase(PHASE_DONE, "handback ACK received")
        elif elapsed >= args.handback_mode_timeout:
            with mission_lock:
                mission_state["handback_done"] = True
            if args.live_fly:
                print(f"[HANDBACK WARN] no DO_SET_MODE ACK in {elapsed:.2f}s; "
                      f"continuing to PHASE_DONE", flush=True)
            _transition_phase(PHASE_DONE, "handback timeout / BLANK")

    # ========================================================================
    # Threads
    # ========================================================================

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
        # Drone-state-machine local state
        locked_on_target = False
        lock_frame_idx = -1
        hold_confirm_count = 0
        altitude_confirm_count = 0
        final_hold_engaged = False
        sim_target_cy = None
        prev_state = None
        # Anti-jitter state — only active in PHASE_DRONE_POSITIONING / HANDOFF_WAIT.
        ema_cx = None
        ema_cy = None
        yaw_aligned_latched = False
        out_of_align_streak = 0
        no_target_streak = 0
        run_start = time.perf_counter()
        # Gimbal frame counter (for log signature)
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

            now = time.perf_counter()
            dt = clamp(now - sim_state["last_tick"], 0.001, 0.2)
            sim_state["last_tick"] = now

            infer_frame = apply_preproc(frame, args)
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

            out = [d for d in filtered
                   if float(d.get("confidence", 0.0)) >= args.min_track_confidence]

            count += 1
            if count % 30 == 0:
                print(f"infer_frames={count} dets={len(out)}", flush=True)

            with det_lock:
                latest_detections[0] = out

            frame_idx += 1
            gimbal_frame_idx += 1

            selected = None
            if out:
                selected = max(out, key=lambda d: float(d.get("confidence", 0.0)))

            # Read current mission phase under lock
            with mission_lock:
                phase = mission_state["phase"]

            # ================================================================
            # PHASE-CONDITIONAL BRANCH
            # ================================================================

            # --- Distance reading (used by drone phases) ---
            if args.simulate_distance:
                lidar_cm = sim_state["lidar_cm"]
                lidar_source = "sim"
            else:
                cm, age = drone_link.get_distance_cm()
                if cm is None or (age is not None and age > 2.0):
                    lidar_cm = (sim_state["lidar_cm"] if cm is None else cm)
                    lidar_source = "stale"
                else:
                    lidar_cm = cm
                    lidar_source = "sensor"
                sim_state["lidar_cm"] = lidar_cm

            frame_h, frame_w = frame.shape[:2]
            fx = frame_w / 2.0
            fy = frame_h / 2.0

            # Task 2 compliance gate (PHASE_DRONE_POSITIONING only, first valid reading)
            if (phase == PHASE_DRONE_POSITIONING
                    and not mission_state["first_lidar_checked"]
                    and lidar_source == "sensor"
                    and lidar_cm is not None
                    and args.min_start_distance_cm > 0.0):
                if lidar_cm < args.min_start_distance_cm:
                    print(f"\n[COMPLIANCE FAIL] First DISTANCE_SENSOR reading "
                          f"{lidar_cm:.1f}cm < --min-start-distance-cm="
                          f"{args.min_start_distance_cm:.0f}cm.\n"
                          f"Task 2 §5.2.4 requires the autonomous approach to "
                          f"start from >2 m. Re-position the drone farther from "
                          f"the target and retry.\n", flush=True)
                    stop_event.set()
                    sys.exit(EXIT_COMPLIANCE_FAIL)
                print(f"[COMPLIANCE OK] start distance {lidar_cm:.1f}cm "
                      f">= {args.min_start_distance_cm:.0f}cm "
                      f"(Task 2 >2m criterion satisfied)", flush=True)
                with mission_lock:
                    mission_state["first_lidar_checked"] = True

            # Reset per-frame outputs (drone)
            state = STATE_NO_TARGET
            move_label = "IDLE"
            displacement_label = "CENTERED"
            err_x = 0.0
            err_y = 0.0
            vx = 0.0
            yaw_rate = 0.0
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

            # Bounds + targets
            low_bound = args.target_distance_cm - args.distance_tolerance_cm
            high_bound = args.target_distance_cm + args.distance_tolerance_cm
            center_low = args.target_distance_cm - args.distance_center_band_cm
            center_high = args.target_distance_cm + args.distance_center_band_cm
            dist_error_cm = lidar_cm - args.target_distance_cm
            target_y_px = args.altitude_target_y_ratio * float(frame_h)

            # ----------------------------------------------------------------
            # Branch A: drone is positioning (state machine active)
            # ----------------------------------------------------------------
            if phase in (PHASE_DRONE_POSITIONING, PHASE_HANDOFF_WAIT):
                if selected is None:
                    no_target_streak += 1
                    dropout_grace = no_target_streak < max(1, args.no_target_dropout_frames)
                    if locked_on_target:
                        state = STATE_LOCKED
                        move_label = "LOCKED_NO_TARGET"
                        displacement_label = "LOCKED_NO_TARGET"
                        altitude_confirm_count = 0
                        if abs(dist_error_cm) > args.distance_center_band_cm:
                            vx = distance_vx_command(
                                lidar_cm, args.target_distance_cm,
                                args.lock_forward_gain, args.max_vx,
                                args.min_distance_correct_vx,
                            )
                            move_label = "LOCKED_DIST_CORRECT"
                    else:
                        state = STATE_NO_TARGET
                        if not dropout_grace:
                            hold_confirm_count = 0
                            altitude_confirm_count = 0
                            final_hold_engaged = False
                            sim_target_cy = None
                            ema_cx = None
                            ema_cy = None
                            if yaw_aligned_latched:
                                print(f"[YAW HYST] released (NO_TARGET streak "
                                      f"{no_target_streak} >= "
                                      f"{args.no_target_dropout_frames})", flush=True)
                            yaw_aligned_latched = False
                            out_of_align_streak = 0
                        # else: brief flicker — keep counters / EMA / latch.
                else:
                    no_target_streak = 0
                    (x1, y1), (x2, y2) = selected["bbox"]
                    raw_cx = (x1 + x2) / 2.0
                    raw_cy = (y1 + y2) / 2.0
                    # EMA smoothing on bbox center to kill 1-frame wobble.
                    if ema_cx is None or ema_cy is None:
                        ema_cx = raw_cx
                        ema_cy = raw_cy
                    else:
                        a = clamp(args.bbox_ema_alpha, 0.0, 1.0)
                        ema_cx = a * raw_cx + (1.0 - a) * ema_cx
                        ema_cy = a * raw_cy + (1.0 - a) * ema_cy
                    cx = ema_cx
                    cy = ema_cy
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

                    displacement_label = direction_label(err_x, err_y, args.drone_deadband)
                    yaw_label = yaw_direction_label(err_x, args.drone_deadband)
                    # Hysteresis latch — see drone_auto for full rationale.
                    entry_db = args.drone_deadband
                    exit_db = args.drone_deadband * max(1.0, args.yaw_hysteresis_ratio)
                    _prev_latch = yaw_aligned_latched
                    if yaw_aligned_latched:
                        if abs(err_x) > exit_db:
                            out_of_align_streak += 1
                            if out_of_align_streak >= max(1, args.yaw_align_dropout_frames):
                                yaw_aligned_latched = False
                                out_of_align_streak = 0
                        else:
                            out_of_align_streak = 0
                    else:
                        if abs(err_x) <= entry_db:
                            yaw_aligned_latched = True
                            out_of_align_streak = 0
                    if _prev_latch != yaw_aligned_latched:
                        print(f"[YAW HYST] "
                              f"{'LATCHED' if yaw_aligned_latched else 'RELEASED'} "
                              f"err_x={err_x:+.3f} entry_db={entry_db:.3f} "
                              f"exit_db={exit_db:.3f}", flush=True)
                    yaw_aligned = yaw_aligned_latched
                    in_distance_window = low_bound <= lidar_cm <= high_bound

                    if locked_on_target:
                        state = STATE_LOCKED
                        lock_deadband = args.drone_deadband * args.lock_deadband_scale
                        move_label = "LOCKED_STEADY"
                        if abs(err_x) > lock_deadband:
                            yaw_rate = clamp(args.lock_yaw_gain * err_x,
                                             -args.max_yaw_rate, args.max_yaw_rate)
                            move_label = "LOCKED_YAW_CORRECT"
                        if abs(dist_error_cm) > args.distance_center_band_cm:
                            vx = distance_vx_command(
                                lidar_cm, args.target_distance_cm,
                                args.lock_forward_gain, args.max_vx,
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
                            yaw_rate = clamp(args.drone_yaw_gain * err_x,
                                             -args.max_yaw_rate, args.max_yaw_rate)
                            move_label = yaw_label
                        else:
                            if lidar_cm > high_bound:
                                hold_confirm_count = 0
                                state = STATE_APPROACH
                                vx = distance_vx_command(
                                    lidar_cm, args.target_distance_cm,
                                    args.forward_gain, args.max_vx,
                                    args.min_distance_correct_vx,
                                )
                                move_label = "FORWARD"
                            elif lidar_cm < low_bound:
                                hold_confirm_count = 0
                                state = STATE_APPROACH
                                vx = distance_vx_command(
                                    lidar_cm, args.target_distance_cm,
                                    args.forward_gain, args.max_vx,
                                    args.min_distance_correct_vx,
                                )
                                move_label = "BACKWARD"
                            else:
                                state = STATE_HOLD
                                move_label = "HOLD"
                                hold_confirm_count += 1
                                if abs(dist_error_cm) > args.distance_center_band_cm:
                                    vx = distance_vx_command(
                                        lidar_cm, args.target_distance_cm,
                                        args.hold_forward_gain, args.max_vx,
                                        args.hold_min_distance_correct_vx,
                                    )
                                    if vx > 0.0:
                                        move_label = "HOLD_TRIM_FORWARD"
                                    elif vx < 0.0:
                                        move_label = "HOLD_TRIM_BACKWARD"

                        if (args.lock_after_approach
                                and yaw_aligned
                                and in_distance_window
                                and hold_confirm_count >= args.lock_confirm_frames):
                            locked_on_target = True
                            lock_frame_idx = frame_idx
                            state = STATE_LOCKED
                            move_label = "LOCKED_STEADY"
                            altitude_confirm_count = 0
                            final_hold_engaged = False
                            print(f"[LOCK ENGAGED] hold_frames={hold_confirm_count} "
                                  f"lidar={lidar_cm:.1f}cm err_x={err_x:+.3f}",
                                  flush=True)

                # Publish velocity for the drone tx loop
                drone_link.set_velocity(vx, 0.0, vz, yaw_rate)

                # Gimbal is STATIC during drone positioning — do NOT update wished_*.
                # The gimbal tx loop will heartbeat the startup setpoint.

                # Simulated LiDAR / altitude dynamics
                if args.simulate_distance:
                    disturbance_vx = args.sim_drift_vx_mps
                    if args.sim_drift_jitter_mps > 0.0:
                        disturbance_vx += random.uniform(
                            -args.sim_drift_jitter_mps, args.sim_drift_jitter_mps)
                    net_vx = vx + disturbance_vx
                    sim_state["lidar_cm"] += (
                        -net_vx * dt * 100.0 * args.sim_lidar_approach_factor)
                    if args.sim_lidar_noise_cm > 0.0:
                        sim_state["lidar_cm"] += random.uniform(
                            -args.sim_lidar_noise_cm, args.sim_lidar_noise_cm)
                    sim_state["lidar_cm"] = clamp(
                        sim_state["lidar_cm"], args.sim_lidar_min_cm,
                        args.sim_lidar_max_cm)
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

                # --- Mission phase transitions for drone phases ---
                if phase == PHASE_DRONE_POSITIONING and state == STATE_FINAL_HOLD:
                    with mission_lock:
                        mission_state["final_hold_streak"] = 1
                    _transition_phase(PHASE_HANDOFF_WAIT, "drone reached FINAL_HOLD")
                elif phase == PHASE_HANDOFF_WAIT:
                    if state == STATE_FINAL_HOLD:
                        with mission_lock:
                            mission_state["final_hold_streak"] += 1
                            streak = mission_state["final_hold_streak"]
                        if streak >= args.handoff_confirm_frames:
                            _transition_phase(
                                PHASE_GIMBAL_TRACKING,
                                f"FINAL_HOLD stable for {args.handoff_confirm_frames} frames")
                            with mission_lock:
                                mission_state["handoff_at"] = time.monotonic()
                    else:
                        with mission_lock:
                            prev_streak = mission_state["final_hold_streak"]
                            mission_state["final_hold_streak"] = 0
                        if prev_streak > 0:
                            print(f"[HANDOFF RESET] state regressed to {state}; "
                                  f"streak {prev_streak} -> 0", flush=True)

            # ----------------------------------------------------------------
            # Branch B: gimbal tracking / firing (drone freezes)
            # ----------------------------------------------------------------
            elif phase in (PHASE_GIMBAL_TRACKING, PHASE_FIRING):
                # FREEZE the drone — continuous zero-velocity setpoints
                drone_link.set_velocity(0.0, 0.0, 0.0, 0.0)
                state = STATE_FINAL_HOLD  # for OSD continuity
                move_label = "FROZEN"
                lidar_source = lidar_source  # passthrough

                # Gimbal tracking
                if selected is not None:
                    (x1, y1), (x2, y2) = selected["bbox"]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    err_x = (cx - fx) / max(1.0, fx)
                    err_y = (cy - fy) / max(1.0, fy)
                    target_confidence = float(selected.get("confidence", 0.0))
                    tcx = int(round(cx))
                    tcy = int(round(cy))
                    move_label_gimbal = direction_label(err_x, err_y, args.gimbal_deadband)
                    displacement_label = move_label_gimbal

                    cur_pitch, cur_yaw = gimbal_link.get_current()
                    if move_label_gimbal != "CENTERED":
                        wished_yaw = clamp(cur_yaw + args.gimbal_yaw_gain * err_x,
                                           GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
                        wished_pitch = clamp(cur_pitch - args.pitch_gain * err_y,
                                             GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
                        gimbal_link.set_wished(wished_pitch, wished_yaw)
                    centered = (move_label_gimbal == "CENTERED")
                else:
                    centered = False

                # Advance the discharge state machine
                prev_fire_phase_local = fire_state["phase"]
                fire_advance(centered=centered)
                new_fire_phase_local = fire_state["phase"]

                # Mission transitions out of GIMBAL_TRACKING / FIRING
                if phase == PHASE_GIMBAL_TRACKING and new_fire_phase_local == "FIRING":
                    with mission_lock:
                        mission_state["fire_started_at"] = time.monotonic()
                    _transition_phase(PHASE_FIRING,
                                      "discharge state machine entered FIRING")
                elif (phase == PHASE_FIRING
                      and prev_fire_phase_local == "FIRING"
                      and new_fire_phase_local == "COOLDOWN"):
                    # Burst complete — go to PHASE_VERIFY (or skip to HANDBACK)
                    if args.no_photo_capture:
                        _transition_phase(
                            PHASE_HANDBACK,
                            f"discharge complete ({args.fire_period:.1f}s); "
                            f"--no-photo-capture")
                        do_handback_once()
                    else:
                        with mission_lock:
                            mission_state["verify_started_at"] = time.monotonic()
                            mission_state["verify_frames_captured"] = 0
                            mission_state["verify_best_frame"] = None
                            mission_state["verify_done"] = False
                        _transition_phase(
                            PHASE_VERIFY,
                            f"discharge complete ({args.fire_period:.1f}s); "
                            f"capturing photos")

            # ----------------------------------------------------------------
            # Branch C: PHASE_VERIFY — capture photos
            # ----------------------------------------------------------------
            elif phase == PHASE_VERIFY:
                # Drone still frozen
                drone_link.set_velocity(0.0, 0.0, 0.0, 0.0)
                state = STATE_FINAL_HOLD
                move_label = "VERIFY"

                # Continue light gimbal tracking so camera stays on the (now-blue) target
                if selected is not None:
                    (x1, y1), (x2, y2) = selected["bbox"]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    err_x = (cx - fx) / max(1.0, fx)
                    err_y = (cy - fy) / max(1.0, fy)
                    target_confidence = float(selected.get("confidence", 0.0))
                    tcx = int(round(cx))
                    tcy = int(round(cy))
                    move_label_gimbal = direction_label(err_x, err_y, args.gimbal_deadband)
                    displacement_label = move_label_gimbal
                    if move_label_gimbal != "CENTERED":
                        cur_pitch, cur_yaw = gimbal_link.get_current()
                        wished_yaw = clamp(cur_yaw + args.gimbal_yaw_gain * err_x,
                                           GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
                        wished_pitch = clamp(cur_pitch - args.pitch_gain * err_y,
                                             GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
                        gimbal_link.set_wished(wished_pitch, wished_yaw)

                # Capture frames
                do_verify_capture(frame, selected, frame_w, frame_h)

                with mission_lock:
                    done = mission_state["verify_done"]
                if done:
                    _transition_phase(PHASE_HANDBACK, "photo captured + saved")
                    do_handback_once()

            # ----------------------------------------------------------------
            # Branch D: PHASE_HANDBACK — drone halted, poll for ACK
            # ----------------------------------------------------------------
            elif phase == PHASE_HANDBACK:
                drone_link.set_velocity(0.0, 0.0, 0.0, 0.0)
                state = STATE_FINAL_HOLD
                move_label = "HANDBACK"
                do_handback_once()

            # ----------------------------------------------------------------
            # Branch E: PHASE_DONE — wind down
            # ----------------------------------------------------------------
            elif phase == PHASE_DONE:
                drone_link.set_velocity(0.0, 0.0, 0.0, 0.0)
                stop_event.set()
                break

            # ================================================================
            # Distance status + overlay publish + per-frame log
            # ================================================================

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

            # Real velocity feedback
            vfr_gs, _, vfr_climb, _, _, vfr_age = drone_link.get_vfr_hud()
            actual_vx = vfr_gs if (vfr_age is not None and vfr_age < 2.0) else None
            actual_vz = vfr_climb if (vfr_age is not None and vfr_age < 2.0) else None
            if actual_vz is not None:
                actual_vz = -actual_vz

            # Current gimbal state for OSD
            gp, gy = gimbal_link.get_current()
            gwp, gwy = gimbal_link.get_wished()
            fire_phase_now = fire_state["phase"]
            fire_elapsed = 0.0
            if fire_phase_now in ("FIRING", "COOLDOWN") and fire_state["fire_send_time"] > 0:
                fire_elapsed = time.monotonic() - fire_state["fire_send_time"]

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
                overlay_state["gimbal_pitch"] = gp
                overlay_state["gimbal_yaw"] = gy
                overlay_state["gimbal_wished_pitch"] = gwp
                overlay_state["gimbal_wished_yaw"] = gwy
                overlay_state["fire_phase"] = fire_phase_now
                overlay_state["fire_elapsed"] = fire_elapsed
                overlay_state["mission_phase"] = phase
                with mission_lock:
                    overlay_state["final_hold_streak"] = mission_state["final_hold_streak"]
                    overlay_state["verify_frames_captured"] = mission_state["verify_frames_captured"]
                    overlay_state["verify_photo_path"] = mission_state["verify_photo_path"]

            # --- State transition log (drone state machine) ---
            if state != prev_state:
                elapsed = time.perf_counter() - run_start
                reason_bits = []
                if selected is None:
                    reason_bits.append("no detection in frame")
                else:
                    reason_bits.append(f"target conf={target_confidence:.2f}")
                    reason_bits.append(f"err_x={err_x:+.3f}")
                reason_bits.append(f"lidar={lidar_cm:.1f}cm[{dist_status}]")
                arrow = f"{prev_state or 'INIT'} -> {state}"
                print(f"[STATE {elapsed:6.2f}s] {arrow:38s}  why: " +
                      " | ".join(reason_bits), flush=True)
                prev_state = state

            # --- Per-frame log line ---
            mode_str, _ = drone_link.get_mode()
            mode_tag = mode_str if mode_str else "NO_HB"
            drone_tx_tag = "LIVE" if args.live_fly else "BLANK"
            fire_tx_tag = "DISABLED" if args.no_fire else ("LIVE" if args.live_fire else "BLANK")

            sel_str = "no_det"
            if selected is not None:
                sel_str = f"conf={target_confidence:.3f}"

            print(
                f"[F{frame_idx:06d} M={phase}] "
                f"drone={state}/{move_label} "
                f"gimbal=cur(y{gy:+.2f},p{gp:+.2f}) wished(y{gwy:+.2f},p{gwp:+.2f}) "
                f"fire={fire_phase_now} "
                f"{sel_str} "
                f"lidar={lidar_cm:.1f}cm({lidar_source}) "
                f"dist_err={dist_error_cm:+.1f}cm "
                f"CMD vx={vx:+.3f} vz={vz:+.3f} yaw={yaw_rate:+.2f}deg/s "
                f"mode={mode_tag} tx={drone_tx_tag}/{fire_tx_tag} "
                f"target={args.team_name}/{args.target_number}",
                flush=True,
            )

    # --- Thread spawn (rx first so initial telemetry is captured before tx gates) ---
    t_shared_rx = threading.Thread(
        target=shared_rx_loop,
        args=(master, drone_link, gimbal_link, _on_handback_mode_ack, stop_event),
        daemon=True,
    )
    t_drone_tx = threading.Thread(target=drone_link.run_tx_loop,
                                  args=(stop_event,), daemon=True)
    t_gimbal_tx = threading.Thread(target=gimbal_link.run_tx_loop,
                                   args=(stop_event,), daemon=True)
    t_cap = threading.Thread(target=capture_thread, daemon=True)
    t_inf = threading.Thread(target=inference_thread, daemon=True)

    if master is not None:
        t_shared_rx.start()
    t_drone_tx.start()
    t_gimbal_tx.start()
    t_cap.start()
    t_inf.start()

    frame_interval = 1.0 / FPS

    # --- Mission OSD color palette ---
    def _phase_color(phase: str):
        return {
            PHASE_DRONE_POSITIONING: (255, 200, 0),   # cyan-ish
            PHASE_HANDOFF_WAIT:      (0, 255, 255),   # yellow
            PHASE_GIMBAL_TRACKING:   (255, 0, 255),   # magenta
            PHASE_FIRING:            (0, 0, 255),     # red
            PHASE_VERIFY:            (255, 255, 255), # white
            PHASE_HANDBACK:          (0, 255, 0),     # green
            PHASE_DONE:              (0, 255, 0),
        }.get(phase, (200, 200, 200))

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
                    # --- Detection boxes ---
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

                    if selected is not None:
                        tcx = int(o["tcx"])
                        tcy = int(o["tcy"])
                        cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)
                        yaw_end_x = int(round(float(fx_i) + o["yaw_vec_dx_px"]))
                        cv2.arrowedLine(frame, (fx_i, fy_i), (yaw_end_x, fy_i),
                                        (0, 255, 255), 2, tipLength=0.18)
                        cv2.line(frame, (yaw_end_x, fy_i - 8), (yaw_end_x, fy_i + 8),
                                 (0, 255, 255), 2)
                        alt_target_y = int(round(o["target_y_px"]))
                        cv2.line(frame, (0, alt_target_y), (frame_w - 1, alt_target_y),
                                 (120, 80, 255), 1)
                        cv2.arrowedLine(frame, (tcx, tcy), (tcx, alt_target_y),
                                        (255, 170, 0), 2, tipLength=0.18)
                        cv2.circle(frame, (tcx, alt_target_y), 5, (255, 170, 0), -1)

                    # --- Mission banner (top center) ---
                    mp = o.get("mission_phase", PHASE_DRONE_POSITIONING)
                    elapsed_s = time.monotonic() - run_start_mono
                    banner = (f"MISSION: {mp} | TARGET: {args.team_name}/"
                              f"#{args.target_number} | t={elapsed_s:.1f}s")
                    cv2.putText(frame, banner, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                _phase_color(mp), 2, cv2.LINE_AA)

                    # --- Left column: drone state info ---
                    cv2.putText(
                        frame,
                        f"state={o['state']} action={o['move_label']} "
                        f"disp={o['displacement_label']} "
                        f"err=({o['err_x']:+.3f},{o['err_y']:+.3f})",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255),
                        2, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        f"lidar={o['lidar_cm']:.1f}cm({o['lidar_source']}) "
                        f"({o['dist_status']}) "
                        f"target={args.target_distance_cm:.0f}"
                        f"+-{args.distance_tolerance_cm:.0f} "
                        f"dist_err={o['dist_error_cm']:+.1f}cm "
                        f"cmd[yaw,vx,vz]=({o['yaw_rate']:+.1f},"
                        f"{o['vx']:+.2f},{o['vz']:+.2f})",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                        (255, 255, 255), 1, cv2.LINE_AA)
                    lock_color = ((80, 255, 80) if o["final_hold_status"] == "ON"
                                  else ((120, 230, 255) if o["lock_status"] == "ON"
                                        else (200, 200, 200)))
                    cv2.putText(
                        frame,
                        f"lock={o['lock_status']} "
                        f"hold={o['hold_confirm_count']}/{args.lock_confirm_frames} "
                        f"alt={o['altitude_confirm_count']}/{args.altitude_lock_confirm_frames} "
                        f"final={o['final_hold_status']} "
                        f"final_streak={o['final_hold_streak']}/{args.handoff_confirm_frames}",
                        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, lock_color,
                        2, cv2.LINE_AA)

                    # Mode + tx mode line
                    mode_str, _ = drone_link.get_mode()
                    mode_display = mode_str if mode_str else "NO_HEARTBEAT"
                    mode_color = (0, 255, 0) if mode_display == "GUIDED" else (0, 120, 255)
                    cv2.putText(frame, f"mode={mode_display}", (10, 132),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color,
                                2, cv2.LINE_AA)
                    tx_display = "LIVE-FLY" if args.live_fly else "BLANK"
                    tx_color = (0, 80, 255) if args.live_fly else (200, 200, 0)
                    cv2.putText(frame, f"tx={tx_display}", (240, 132),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tx_color,
                                2, cv2.LINE_AA)

                    # Feedback (VFR_HUD)
                    if o.get("actual_vx") is not None:
                        cv2.putText(
                            frame,
                            f"feedback: gs={o['actual_vx']:.2f}m/s "
                            f"vz_body={o['actual_vz']:+.2f}m/s",
                            (10, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (200, 200, 255), 1, cv2.LINE_AA)

                    # --- Right column: gimbal state ---
                    rx = max(10, frame_w - 420)
                    cv2.putText(
                        frame,
                        f"gimbal cur: y={o['gimbal_yaw']:+.2f} p={o['gimbal_pitch']:+.2f}",
                        (rx, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 255, 180), 2, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        f"gimbal wish: y={o['gimbal_wished_yaw']:+.2f} "
                        f"p={o['gimbal_wished_pitch']:+.2f}",
                        (rx, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 230, 255), 1, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        f"slew yaw<={args.max_slew_rate_yaw:.1f}dps "
                        f"pitch<={args.max_slew_rate_pitch:.1f}dps",
                        (rx, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

                    # --- Bottom center: discharge / verify state ---
                    fp = o.get("fire_phase", "IDLE")
                    fe = o.get("fire_elapsed", 0.0)
                    fire_label = f"DISCHARGE: {fp}"
                    if fp == "FIRING":
                        fire_label = (f"FIRING ({'LIVE' if args.live_fire else 'BLANK'}) "
                                      f"t={fe:.2f}s")
                    if mp == PHASE_VERIFY:
                        fire_label = (f"VERIFY: capturing "
                                      f"{o['verify_frames_captured']}/"
                                      f"{args.capture_frame_count}")
                    elif mp == PHASE_HANDBACK or mp == PHASE_DONE:
                        if o.get("verify_photo_path"):
                            fire_label = f"PHOTO: {pathlib.Path(o['verify_photo_path']).name}"
                        else:
                            fire_label = f"HANDBACK -> {args.handback_mode}"
                    fire_color = (0, 255, 0) if fp == "FIRING" else (255, 255, 255)
                    cv2.putText(frame, fire_label, (10, frame_h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fire_color,
                                2, cv2.LINE_AA)

                    # --- Bottom left: detections + frame count ---
                    cv2.putText(
                        frame,
                        f"detections={len(detections)} "
                        f"raw={o['raw_detection_count']} frame={o['frame_idx']}",
                        (10, frame_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
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

        # Best-effort handback if mission didn't reach PHASE_DONE
        try:
            with mission_lock:
                cur_phase = mission_state["phase"]
            if cur_phase not in (PHASE_DONE, PHASE_HANDBACK):
                _transition_phase(PHASE_HANDBACK, "teardown — best-effort")
                do_handback_once()
                # Wait briefly for ACK or timeout
                end = time.monotonic() + args.handback_mode_timeout
                while time.monotonic() < end:
                    do_handback_once()
                    with mission_lock:
                        if mission_state["handback_done"]:
                            break
                    time.sleep(0.05)
        except Exception as e:
            print(f"[TEARDOWN WARN] handback failed: {e}", flush=True)

        # Final safety halt + relay OFF (idempotent re-runs are safe)
        if master is not None and args.live_fly:
            try:
                drone_link.send_halt()
            except Exception as e:
                print(f"[HALT] WARN: halt failed on exit: {e}", flush=True)
        if not args.no_fire and master is not None:
            try:
                send_relay(master, args.fire_relay, 0)
                print(f"[FIRE] safety: relay {args.fire_relay} forced OFF on exit",
                      flush=True)
            except Exception as e:
                print(f"[FIRE] WARN: safety relay OFF failed on exit: {e}", flush=True)

        t_inf.join(timeout=2.0)
        t_drone_tx.join(timeout=2.0)
        t_gimbal_tx.join(timeout=2.0)
        t_cap.join(timeout=2.0)
        if master is not None:
            t_shared_rx.join(timeout=2.0)

        cap.release()
        if writer is not None:
            writer.release()
        if master is not None:
            try:
                master.close()
            except Exception:
                pass

        # Final summary
        with mission_lock:
            photo = mission_state["verify_photo_path"]
        print("=" * 78, flush=True)
        print(f"[MISSION DONE] team={args.team_name} target={args.target_number}",
              flush=True)
        if photo:
            print(f"[MISSION DONE] photo={photo}", flush=True)
            print("[MISSION DONE] Upload to team Google Drive folder and visually "
                  "verify BLUE color before declaring to judges.", flush=True)
        else:
            print("[MISSION DONE] no photo captured", flush=True)
        print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
