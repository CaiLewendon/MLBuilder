import argparse
import math
import random
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_PATH = ROOT / "export" / "project1_prod_saved_model" / "project1_prod_float16.tflite"

STATE_NO_TARGET = "NO_TARGET"
STATE_CENTERING = "CENTERING"
STATE_APPROACH = "APPROACH"
STATE_HOLD = "HOLD"
STATE_LOCKED = "LOCKED_HOLD"
STATE_ALTITUDE_ADJUST = "ALTITUDE_ADJUST"
STATE_FINAL_HOLD = "FINAL_HOLD"


def center_crop(frame, ratio: float):
    h, w = frame.shape[:2]
    crop_w = max(1, int(w * ratio))
    crop_h = max(1, int(h * ratio))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return frame[y0 : y0 + crop_h, x0 : x0 + crop_w], x0, y0


def remap_detections(detections, x_off: int, y_off: int):
    remapped = []
    for detection in detections:
        (x1, y1), (x2, y2) = detection["bbox"]
        d = dict(detection)
        d["bbox"] = ((int(x1) + x_off, int(y1) + y_off), (int(x2) + x_off, int(y2) + y_off))
        remapped.append(d)
    return remapped


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [label for label in labels if label]


def label_for_class_id(class_id: int, labels: list[str]) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


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
    distance_error = current_distance_cm - target_distance_cm
    if distance_error == 0.0:
        return 0.0
    normalized_error = abs(distance_error) / max(1.0, target_distance_cm)
    vx_cmd = clamp(gain * normalized_error, 0.0, max_vx)
    if vx_cmd > 0.0:
        vx_cmd = max(vx_cmd, min_vx)
    return vx_cmd if distance_error > 0.0 else -vx_cmd


def main():
    parser = argparse.ArgumentParser(
        prog="tflive_drone_sim",
        description="TF Live Infrence with print-only drone centering/approach simulation",
    )

    parser.add_argument(
        "model",
        nargs="?",
        default=str(DEFAULT_MODEL_PATH),
        help=f"TFlite model to run (default: {DEFAULT_MODEL_PATH})",
        type=str,
    )
    parser.add_argument(
        "--video",
        "-v",
        help="What video source to use (defaults to default camera)",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--nms",
        "-n",
        help="Apply NMS post processing (add if not compiled in model)",
        action="store_false",
    )
    parser.add_argument(
        "--tpu",
        "-t",
        help="Attempt to delegate to Coral TPU (Will fall back to cpu if not edge compiled)",
        action="store_true",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        help="Minimum confidence threshold for detections",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--min-track-confidence",
        type=float,
        default=0.35,
        help="Secondary confidence gate for control/overlay filtering",
    )
    parser.add_argument(
        "--center-crop-pass",
        action="store_true",
        default=True,
        help="Run a second detection pass on a center crop and merge detections",
    )
    parser.add_argument(
        "--no-center-crop-pass",
        action="store_false",
        dest="center_crop_pass",
        help="Disable the second center-crop detection pass",
    )
    parser.add_argument(
        "--center-crop-ratio",
        type=float,
        default=0.5,
        help="Center crop size ratio for second pass (0<ratio<=1.0)",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.08,
        help="Normalized deadband around image center before issuing movement updates",
    )
    parser.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=78.0,
        help="Camera horizontal FOV in degrees for yaw-to-center estimate",
    )
    parser.add_argument(
        "--yaw-gain",
        type=float,
        default=35.0,
        help="Yaw-rate gain in deg/s at full-scale horizontal error",
    )
    parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=25.0,
        help="Maximum absolute yaw-rate command in deg/s",
    )
    parser.add_argument(
        "--forward-gain",
        type=float,
        default=0.80,
        help="Distance-control gain for vx command while adjusting stand-off distance",
    )
    parser.add_argument(
        "--max-vx",
        type=float,
        default=0.55,
        help="Maximum forward velocity (m/s)",
    )
    parser.add_argument(
        "--min-distance-correct-vx",
        type=float,
        default=0.08,
        help="Minimum |vx| used when distance is outside tolerance (helps visible backward/forward correction)",
    )
    parser.add_argument(
        "--target-distance-cm",
        type=float,
        default=200.0,
        help="Target stand-off distance from target plane in cm",
    )
    parser.add_argument(
        "--distance-tolerance-cm",
        type=float,
        default=15.0,
        help="Allowed stand-off tolerance in cm",
    )
    parser.add_argument(
        "--distance-center-band-cm",
        type=float,
        default=2.0,
        help="Tighter inner band around target distance for active trim control",
    )
    parser.add_argument(
        "--hold-forward-gain",
        type=float,
        default=0.35,
        help="Distance-control gain used while in HOLD and inside tolerance window",
    )
    parser.add_argument(
        "--hold-min-distance-correct-vx",
        type=float,
        default=0.03,
        help="Minimum |vx| used for distance trim while in HOLD",
    )
    parser.add_argument(
        "--lock-after-approach",
        action="store_true",
        default=True,
        help="Lock mission phase once aligned and at target distance, then actively stabilize there",
    )
    parser.add_argument(
        "--no-lock-after-approach",
        action="store_false",
        dest="lock_after_approach",
        help="Disable lock-latch behavior after reaching target",
    )
    parser.add_argument(
        "--lock-confirm-frames",
        type=int,
        default=8,
        help="Consecutive frames in hold window required before lock engages",
    )
    parser.add_argument(
        "--lock-yaw-gain",
        type=float,
        default=45.0,
        help="Yaw-rate gain used during LOCKED_HOLD active stabilization",
    )
    parser.add_argument(
        "--lock-forward-gain",
        type=float,
        default=1.00,
        help="Distance-control gain used during LOCKED_HOLD active stabilization",
    )
    parser.add_argument(
        "--lock-deadband-scale",
        type=float,
        default=0.50,
        help="Multiplier on base deadband while in LOCKED_HOLD (smaller=tighter hold)",
    )
    parser.add_argument(
        "--altitude-target-y-ratio",
        type=float,
        default=0.75,
        help="Target vertical position in frame after lock (0.75 puts target in bottom quarter boundary)",
    )
    parser.add_argument(
        "--altitude-deadband",
        type=float,
        default=0.04,
        help="Normalized deadband for altitude placement and hold",
    )
    parser.add_argument(
        "--altitude-gain",
        type=float,
        default=0.55,
        help="Vertical gain for vz command during altitude adjust/hold",
    )
    parser.add_argument(
        "--max-vz",
        type=float,
        default=0.35,
        help="Maximum absolute vertical velocity command (m/s)",
    )
    parser.add_argument(
        "--altitude-lock-confirm-frames",
        type=int,
        default=8,
        help="Consecutive frames in altitude deadband required before FINAL_HOLD",
    )
    parser.add_argument(
        "--sim-lidar-start-cm",
        type=float,
        default=300.0,
        help="Initial simulated forward LiDAR value in cm",
    )
    parser.add_argument(
        "--sim-lidar-noise-cm",
        type=float,
        default=0.5,
        help="Random per-frame simulated LiDAR noise (+/- cm)",
    )
    parser.add_argument(
        "--sim-lidar-min-cm",
        type=float,
        default=40.0,
        help="Minimum simulated LiDAR clamp in cm",
    )
    parser.add_argument(
        "--sim-lidar-max-cm",
        type=float,
        default=1000.0,
        help="Maximum simulated LiDAR clamp in cm",
    )
    parser.add_argument(
        "--sim-lidar-approach-factor",
        type=float,
        default=1.0,
        help="Scale factor applied to vx->distance change in the LiDAR simulation",
    )
    parser.add_argument(
        "--sim-lidar-step-cm",
        type=float,
        default=5.0,
        help="Manual key-adjustment step in cm for simulated LiDAR",
    )
    parser.add_argument(
        "--sim-drift-vx-mps",
        type=float,
        default=0.03,
        help="Constant external disturbance velocity along body-x (+ reduces lidar distance)",
    )
    parser.add_argument(
        "--sim-drift-jitter-mps",
        type=float,
        default=0.06,
        help="Random external disturbance jitter on body-x per frame",
    )
    parser.add_argument(
        "--sim-altitude-response-px-per-m",
        type=float,
        default=180.0,
        help="Pixel-per-meter scale for altitude simulation response",
    )
    parser.add_argument(
        "--sim-altitude-drift-vz-mps",
        type=float,
        default=0.02,
        help="Constant external vertical disturbance in body-z (+down, -up)",
    )
    parser.add_argument(
        "--sim-altitude-drift-jitter-vz-mps",
        type=float,
        default=0.05,
        help="Random external vertical disturbance jitter in body-z per frame",
    )
    parser.add_argument(
        "--sim-altitude-noise-px",
        type=float,
        default=0.3,
        help="Random per-frame visual noise applied to simulated target vertical position",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=str(ROOT / "target_detector_labels.txt"),
        help="Path to labels file (default: target_detector_labels.txt)",
    )

    args = parser.parse_args()
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video

    model_path = Path(args.model).expanduser().resolve()
    use_nms = args.nms
    use_tpu = args.tpu
    tolerance = args.confidence
    crop_pass = args.center_crop_pass
    crop_ratio = args.center_crop_ratio
    labels_path = Path(args.labels).expanduser().resolve()

    if crop_pass and not (0.0 < crop_ratio <= 1.0):
        parser.error("--center-crop-ratio must be > 0.0 and <= 1.0")
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
    if not model_path.is_file():
        parser.error(f"model file not found: {model_path}")
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
    if args.hold_min_distance_correct_vx < 0.0:
        parser.error("--hold-min-distance-correct-vx must be >= 0.0")
    if args.hold_min_distance_correct_vx > args.max_vx:
        parser.error("--hold-min-distance-correct-vx must be <= --max-vx")
    if args.lock_confirm_frames <= 0:
        parser.error("--lock-confirm-frames must be > 0")
    if args.altitude_lock_confirm_frames <= 0:
        parser.error("--altitude-lock-confirm-frames must be > 0")
    if args.lock_deadband_scale <= 0.0:
        parser.error("--lock-deadband-scale must be > 0.0")
    if args.sim_drift_jitter_mps < 0.0:
        parser.error("--sim-drift-jitter-mps must be >= 0.0")
    if args.sim_altitude_response_px_per_m <= 0.0:
        parser.error("--sim-altitude-response-px-per-m must be > 0.0")
    if args.sim_altitude_drift_jitter_vz_mps < 0.0:
        parser.error("--sim-altitude-drift-jitter-vz-mps must be >= 0.0")
    if args.sim_altitude_noise_px < 0.0:
        parser.error("--sim-altitude-noise-px must be >= 0.0")
    if args.sim_lidar_min_cm >= args.sim_lidar_max_cm:
        parser.error("--sim-lidar-min-cm must be < --sim-lidar-max-cm")

    try:
        labels = load_labels(labels_path)
    except FileNotFoundError as e:
        parser.error(str(e))

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        sys.exit(1)

    max_skips = 5
    frame_skip = 0
    frame_idx = 0

    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    m = TFLiteModel(str(model_path))
    m.allocate(tpu=use_tpu)

    sim_lidar_cm = clamp(args.sim_lidar_start_cm, args.sim_lidar_min_cm, args.sim_lidar_max_cm)
    last_tick = time.perf_counter()
    locked_on_target = False
    lock_frame_idx = -1
    hold_confirm_count = 0
    altitude_confirm_count = 0
    final_hold_engaged = False
    sim_target_cy = None

    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Loaded {len(labels)} labels from {labels_path}")
    print("[INFO] Control mode: SIMULATION ONLY (print-only movement intent)")
    print("[INFO] Axis assumptions:")
    print("  image: +x=right, +y=down, center=(frame_w/2, frame_h/2)")
    print("  yaw control: target right -> yaw clockwise (+), target left -> yaw counter-clockwise (-)")
    print("  translation control: once yaw-aligned, +vx=forward, -vx=backward to hold stand-off distance")
    print(
        f"[INFO] Lock mode: {'ENABLED' if args.lock_after_approach else 'DISABLED'} "
        f"(confirm_frames={args.lock_confirm_frames})"
    )
    print(
        f"[INFO] Disturbance sim: drift_vx={args.sim_drift_vx_mps:+.3f}m/s "
        f"jitter={args.sim_drift_jitter_mps:.3f}m/s"
    )
    print(
        f"[INFO] Altitude phase: target_y_ratio={args.altitude_target_y_ratio:.2f} "
        f"alt_deadband={args.altitude_deadband:.3f} confirm_frames={args.altitude_lock_confirm_frames}"
    )
    print("[INFO] Keyboard: '[' or '-' decrease lidar, ']' or '+' increase lidar, 'r' reset lidar, 'q' quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            frame_skip += 1
            print(f"Frame Skip ({frame_skip}/{max_skips}): Attempting Video Reset")
            if frame_skip >= max_skips:
                print("Max Skips Reached, Exiting")
                break
            cap = cv2.VideoCapture(video_source)
            continue
        frame_skip = 0
        frame_idx += 1

        now = time.perf_counter()
        dt = clamp(now - last_tick, 0.001, 0.2)
        last_tick = now

        out = m.detect(frame, nms=use_nms, tol=tolerance)
        if crop_pass:
            cropped, x_off, y_off = center_crop(frame, crop_ratio)
            crop_out = m.detect(cropped, nms=use_nms, tol=tolerance)
            out.extend(remap_detections(crop_out, x_off, y_off))
        raw_detection_count = len(out)
        out = [
            detection
            for detection in out
            if float(detection.get("confidence", 0.0)) >= args.min_track_confidence
        ]

        selected = None
        if out:
            selected = max(out, key=lambda d: d.get("confidence", 0.0))

        for detection in out:
            bbox = detection["bbox"]
            x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
            x_max, y_max = int(bbox[1][0]), int(bbox[1][1])
            class_id = int(detection.get("id", -1))
            class_name = label_for_class_id(class_id, labels)
            confidence = float(detection.get("confidence", 0.0))
            color = (0, 255, 0)
            if detection is selected:
                color = (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                frame,
                f"{class_name} ({class_id}) conf={confidence:.2f}",
                (x_min, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        frame_h, frame_w = frame.shape[:2]
        fx = int(frame_w / 2.0)
        fy = int(frame_h / 2.0)
        cv2.drawMarker(
            frame,
            (fx, fy),
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        cv2.putText(
            frame,
            "CENTER AXIS",
            (fx + 10, fy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

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
        tcx = fx
        tcy = fy
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

        low_bound = args.target_distance_cm - args.distance_tolerance_cm
        high_bound = args.target_distance_cm + args.distance_tolerance_cm
        center_low = args.target_distance_cm - args.distance_center_band_cm
        center_high = args.target_distance_cm + args.distance_center_band_cm
        dist_error_cm = sim_lidar_cm - args.target_distance_cm
        target_y_px = args.altitude_target_y_ratio * float(frame_h)

        alt_error_px = 0.0
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
                        sim_lidar_cm,
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
            in_distance_window = low_bound <= sim_lidar_cm <= high_bound

            if locked_on_target:
                state = STATE_LOCKED
                lock_deadband = args.deadband * args.lock_deadband_scale
                move_label = "LOCKED_STEADY"
                if abs(err_x) > lock_deadband:
                    yaw_rate = clamp(args.lock_yaw_gain * err_x, -args.max_yaw_rate, args.max_yaw_rate)
                    move_label = "LOCKED_YAW_CORRECT"
                if abs(dist_error_cm) > args.distance_center_band_cm:
                    vx = distance_vx_command(
                        sim_lidar_cm,
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
                        vz = clamp(-args.altitude_gain * alt_error_norm, -args.max_vz, args.max_vz)
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
                        vz = clamp(-args.altitude_gain * alt_error_norm, -args.max_vz, args.max_vz)
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
                    yaw_rate = clamp(args.yaw_gain * err_x, -args.max_yaw_rate, args.max_yaw_rate)
                    move_label = yaw_label
                else:
                    if sim_lidar_cm > high_bound:
                        hold_confirm_count = 0
                        state = STATE_APPROACH
                        vx = distance_vx_command(
                            sim_lidar_cm,
                            args.target_distance_cm,
                            args.forward_gain,
                            args.max_vx,
                            args.min_distance_correct_vx,
                        )
                        move_label = "FORWARD"
                    elif sim_lidar_cm < low_bound:
                        hold_confirm_count = 0
                        state = STATE_APPROACH
                        vx = distance_vx_command(
                            sim_lidar_cm,
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
                                sim_lidar_cm,
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

            cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)
            yaw_end_x = int(round(float(fx) + yaw_vec_dx_px))
            cv2.arrowedLine(frame, (fx, fy), (yaw_end_x, fy), (0, 255, 255), 2, tipLength=0.18)
            cv2.line(frame, (yaw_end_x, fy - 8), (yaw_end_x, fy + 8), (0, 255, 255), 2)
            alt_target_y = int(round(target_y_px))
            # Altitude goal guide line (desired target y-position after approach).
            cv2.line(frame, (0, alt_target_y), (frame_w - 1, alt_target_y), (120, 80, 255), 1)
            # Target-point altitude vector: current target point -> desired altitude point.
            cv2.arrowedLine(
                frame,
                (tcx, tcy),
                (tcx, alt_target_y),
                (255, 170, 0),
                2,
                tipLength=0.18,
            )
            cv2.circle(frame, (tcx, alt_target_y), 5, (255, 170, 0), -1)
            cv2.putText(
                frame,
                "ALT TARGET",
                (tcx + 8, max(20, alt_target_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 170, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "ALT GOAL LINE",
                (10, max(20, alt_target_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (120, 80, 255),
                1,
                cv2.LINE_AA,
            )

        # Simulated LiDAR dynamics: commanded motion + external drift/wind disturbance + measurement noise.
        disturbance_vx = args.sim_drift_vx_mps
        if args.sim_drift_jitter_mps > 0.0:
            disturbance_vx += random.uniform(-args.sim_drift_jitter_mps, args.sim_drift_jitter_mps)
        net_vx = vx + disturbance_vx
        sim_lidar_cm += -net_vx * dt * 100.0 * args.sim_lidar_approach_factor
        if args.sim_lidar_noise_cm > 0.0:
            sim_lidar_cm += random.uniform(-args.sim_lidar_noise_cm, args.sim_lidar_noise_cm)
        sim_lidar_cm = clamp(sim_lidar_cm, args.sim_lidar_min_cm, args.sim_lidar_max_cm)

        disturbance_vz = args.sim_altitude_drift_vz_mps
        if args.sim_altitude_drift_jitter_vz_mps > 0.0:
            disturbance_vz += random.uniform(
                -args.sim_altitude_drift_jitter_vz_mps, args.sim_altitude_drift_jitter_vz_mps
            )
        if sim_target_cy is not None:
            net_vz = vz + disturbance_vz
            sim_target_cy += -net_vz * dt * args.sim_altitude_response_px_per_m
            if args.sim_altitude_noise_px > 0.0:
                sim_target_cy += random.uniform(-args.sim_altitude_noise_px, args.sim_altitude_noise_px)
            sim_target_cy = clamp(sim_target_cy, 0.0, float(frame_h - 1))

        dist_error_cm = sim_lidar_cm - args.target_distance_cm
        alt_error_px = target_y_px - control_cy
        alt_error_norm = alt_error_px / max(1.0, float(fy))

        dist_status = "ON_TARGET"
        if sim_lidar_cm > high_bound:
            dist_status = "FAR"
        elif sim_lidar_cm < low_bound:
            dist_status = "CLOSE"
        elif sim_lidar_cm > center_high or sim_lidar_cm < center_low:
            dist_status = "IN_WINDOW_TRIMMING"
        else:
            dist_status = "CENTER_BAND"

        cv2.putText(
            frame,
            f"state={state} action={move_label} disp={displacement_label} err=({err_x:+.3f},{err_y:+.3f})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        lock_status = "ON" if locked_on_target else "OFF"
        final_hold_status = "ON" if final_hold_engaged else "OFF"
        lock_age = 0 if lock_frame_idx < 0 else frame_idx - lock_frame_idx
        cv2.putText(
            frame,
            (
                f"lidar={sim_lidar_cm:.1f}cm ({dist_status}) target={args.target_distance_cm:.0f}"
                f"+-{args.distance_tolerance_cm:.0f} dist_err={dist_error_cm:+.1f}cm "
                f"cmd[yaw_rate,vx,vz]=({yaw_rate:+.1f}deg/s,{vx:+.2f}m/s,{vz:+.2f}m/s)"
            ),
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"lock={lock_status} hold_frames={hold_confirm_count}/{args.lock_confirm_frames} "
                f"alt_frames={altitude_confirm_count}/{args.altitude_lock_confirm_frames} "
                f"final_hold={final_hold_status} lock_age={lock_age} "
                f"disturb[vx,vz]=({disturbance_vx:+.2f},{disturbance_vz:+.2f})m/s"
            ),
            (10, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 255, 80) if final_hold_engaged else ((120, 230, 255) if locked_on_target else (200, 200, 200)),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"yaw_to_center={yaw_to_center_deg:+.2f} deg",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"yaw_vec_px=({yaw_vec_dx_px:+.1f},+0.0) |v|={yaw_vec_mag_px:.1f}px "
                f"bearing_vec_body=(fwd={body_vec_fwd:+.3f},right={body_vec_right:+.3f})"
            ),
            (10, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"alt_to_goal_vec_px=(+0.0,{alt_vec_dy_px:+.1f}) |v|={alt_vec_mag_px:.1f}px "
                f"goal_y={target_y_px:.1f}px alt_err_norm={alt_error_norm:+.3f}"
            ),
            (10, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 180, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"target_offset_ref_px=({disp_dx_px:+.1f},{disp_dy_px:+.1f}) |v|={disp_mag_px:.1f}px",
            (10, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "keys: [/- dec  ]/= inc  r reset  q quit",
            (10, frame_h - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"detections={len(out)} raw={raw_detection_count} frame={frame_idx}",
            (10, frame_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if selected is None:
            print(
                (
                    f"[F{frame_idx:06d}] NO_TARGET state={state} "
                    f"lidar={sim_lidar_cm:.1f}cm det={len(out)}/{raw_detection_count} "
                    f"lock={lock_status} hold_frames={hold_confirm_count}/{args.lock_confirm_frames} "
                    f"alt_frames={altitude_confirm_count}/{args.altitude_lock_confirm_frames} "
                    f"final_hold={final_hold_status} "
                    f"yaw_to_center={yaw_to_center_deg:+.2f}deg "
                    f"yaw_vec_px=({yaw_vec_dx_px:+.1f},+0.0) "
                    f"alt_vec_px=(+0.0,{alt_vec_dy_px:+.1f}) "
                    f"disturb[vx,vz]=({disturbance_vx:+.3f},{disturbance_vz:+.3f})m/s "
                    f"cmd=(yaw_rate={yaw_rate:+.2f}deg/s,vx={vx:+.3f}m/s,vz={vz:+.3f}m/s)"
                )
            )
        else:
            (x1, y1), (x2, y2) = selected["bbox"]
            print(
                (
                    f"[F{frame_idx:06d}] target_bbox=(({int(x1)},{int(y1)}),({int(x2)},{int(y2)})) "
                    f"center=({tcx},{tcy}) err=({err_x:+.3f},{err_y:+.3f}) conf={target_confidence:.3f} "
                    f"det={len(out)}/{raw_detection_count} "
                    f"state={state} action={move_label} disp={displacement_label} "
                    f"lock={lock_status} hold_frames={hold_confirm_count}/{args.lock_confirm_frames} "
                    f"alt_frames={altitude_confirm_count}/{args.altitude_lock_confirm_frames} "
                    f"final_hold={final_hold_status} "
                    f"yaw_to_center={yaw_to_center_deg:+.2f}deg "
                    f"yaw_vec_px=({yaw_vec_dx_px:+.1f},+0.0) "
                    f"alt_vec_px=(+0.0,{alt_vec_dy_px:+.1f}) alt_err_norm={alt_error_norm:+.3f} "
                    f"target_offset_ref_px=({disp_dx_px:+.1f},{disp_dy_px:+.1f}) "
                    f"bearing_vec_body=(fwd={body_vec_fwd:+.3f},right={body_vec_right:+.3f}) "
                    f"lidar={sim_lidar_cm:.1f}cm dist={dist_status} "
                    f"target={args.target_distance_cm:.1f}+-{args.distance_tolerance_cm:.1f} "
                    f"disturb[vx,vz]=({disturbance_vx:+.3f},{disturbance_vz:+.3f})m/s "
                    f"CMD_SIM yaw_rate={yaw_rate:+.2f}deg/s vx={vx:+.3f}m/s vz={vz:+.3f}m/s"
                )
            )

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("]"), ord("="), ord("+")):
            sim_lidar_cm = clamp(
                sim_lidar_cm + args.sim_lidar_step_cm,
                args.sim_lidar_min_cm,
                args.sim_lidar_max_cm,
            )
            print(f"[INPUT] lidar increased -> {sim_lidar_cm:.1f} cm")
        elif key in (ord("["), ord("-"), ord("_")):
            sim_lidar_cm = clamp(
                sim_lidar_cm - args.sim_lidar_step_cm,
                args.sim_lidar_min_cm,
                args.sim_lidar_max_cm,
            )
            print(f"[INPUT] lidar decreased -> {sim_lidar_cm:.1f} cm")
        elif key == ord("r"):
            sim_lidar_cm = clamp(args.sim_lidar_start_cm, args.sim_lidar_min_cm, args.sim_lidar_max_cm)
            print(f"[INPUT] lidar reset -> {sim_lidar_cm:.1f} cm")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
