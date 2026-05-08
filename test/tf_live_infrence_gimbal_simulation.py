import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_PATH = ROOT / "export" / "project1_prod_saved_model" / "project1_prod_float16.tflite"

GIMBAL_YAW_MIN_DEG = -90.0
GIMBAL_YAW_MAX_DEG = 90.0
GIMBAL_PITCH_MIN_DEG = -45.0
GIMBAL_PITCH_MAX_DEG = 45.0
MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_MOUNT_MODE_MAVLINK_TARGETING = 2


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


def main():
    parser = argparse.ArgumentParser(
        prog="tflive_mavlink", description="TF Live Infrence with print-only MAVLink gimbal commands"
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
        "--yaw-gain",
        type=float,
        default=12.0,
        help="Yaw gain in deg per frame at full-scale horizontal error",
    )
    parser.add_argument(
        "--pitch-gain",
        type=float,
        default=10.0,
        help="Pitch gain in deg per frame at full-scale vertical error",
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
    if not model_path.is_file():
        parser.error(f"model file not found: {model_path}")
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
    current_yaw = 0.0
    current_pitch = 0.0

    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    m = TFLiteModel(str(model_path))
    m.allocate(tpu=use_tpu)

    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Loaded {len(labels)} labels from {labels_path}")
    print("[INFO] MAVLink mode: PRINT-ONLY (no connection attempted)")
    print("[INFO] Axis assumptions:")
    print("  image: +x=right, +y=down, center=(frame_w/2, frame_h/2)")
    print("  gimbal yaw (param3): +right/clockwise, -left/counter-clockwise")
    print("  gimbal pitch (param1): +up, -down")
    print("  control: target right -> yaw+, target left -> yaw-, target down -> pitch-, target up -> pitch+")

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

        out = m.detect(frame, nms=use_nms, tol=tolerance)
        if crop_pass:
            cropped, x_off, y_off = center_crop(frame, crop_ratio)
            crop_out = m.detect(cropped, nms=use_nms, tol=tolerance)
            out.extend(remap_detections(crop_out, x_off, y_off))

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
        cv2.putText(
            frame,
            f"detections={len(out)} frame={frame_idx}",
            (10, frame_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if selected is None:
            print(f"[F{frame_idx:06d}] NO_TARGET")
        else:
            (x1, y1), (x2, y2) = selected["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            err_x = (cx - float(fx)) / max(1.0, float(fx))
            err_y = (cy - float(fy)) / max(1.0, float(fy))

            move_label = direction_label(err_x, err_y, args.deadband)
            if move_label != "CENTERED":
                current_yaw += args.yaw_gain * err_x
                current_pitch -= args.pitch_gain * err_y
                current_yaw = clamp(current_yaw, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
                current_pitch = clamp(current_pitch, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)

            tcx = int(round(cx))
            tcy = int(round(cy))
            cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)
            cv2.line(frame, (fx, fy), (tcx, tcy), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{move_label} yaw={current_yaw:.1f} pitch={current_pitch:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            confidence = float(selected.get("confidence", 0.0))
            print(
                (
                    f"[F{frame_idx:06d}] target_bbox=(({int(x1)},{int(y1)}),({int(x2)},{int(y2)})) "
                    f"center=({tcx},{tcy}) err=({err_x:+.3f},{err_y:+.3f}) conf={confidence:.3f} dir={move_label} "
                    f"yaw={current_yaw:.2f} pitch={current_pitch:.2f} "
                    f"CMD_LONG cmd={MAV_CMD_DO_MOUNT_CONTROL} "
                    f"param1={current_pitch:.2f} param2=0.00 param3={current_yaw:.2f} "
                    f"param4=0.00 param5=0.00 param6=0.00 "
                    f"param7={MAV_MOUNT_MODE_MAVLINK_TARGETING}"
                )
            )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
