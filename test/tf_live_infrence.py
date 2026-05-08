import sys
from pathlib import Path

import cv2
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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


def main():
    parser = argparse.ArgumentParser(
        prog="tflive", description="TF Live Infrence Model Test"
    )

    parser.add_argument("model", help="TFlite model to run", type=str)
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
        default=0.25,
    )
    parser.add_argument(
        "--center-crop-pass",
        action="store_true",
        help="Run a second detection pass on a center crop and merge detections",
    )
    parser.add_argument(
        "--center-crop-ratio",
        type=float,
        default=0.5,
        help="Center crop size ratio for second pass (0<ratio<=1.0)",
    )
    parser.add_argument(
        "--log-detections",
        action="store_true",
        help="Print per-frame detection count and bbox coordinates to console",
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
    except:
        video_source = args.video
    model_path = args.model
    use_nms = args.nms
    use_tpu = args.tpu
    tolerance = args.confidence
    crop_pass = args.center_crop_pass
    crop_ratio = args.center_crop_ratio
    log_detections = args.log_detections
    labels_path = Path(args.labels).expanduser().resolve()

    if crop_pass and not (0.0 < crop_ratio <= 1.0):
        parser.error("--center-crop-ratio must be > 0.0 and <= 1.0")

    try:
        labels = load_labels(labels_path)
    except FileNotFoundError as e:
        parser.error(str(e))
    print(f"[INFO] Loaded {len(labels)} labels from {labels_path}")

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    max_skips = 5
    frame_skip = 0

    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    m = TFLiteModel(model_path)
    m.allocate(tpu=use_tpu)

    frame_idx = 0
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

        cv2.putText(
            frame,
            f"detections={len(out)} frame={frame_idx}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for detection in out:
            bbox = detection["bbox"]
            x_min, y_min = bbox[0]
            x_max, y_max = bbox[1]
            class_id = int(detection.get("id", -1))
            class_name = label_for_class_id(class_id, labels)
            confidence = float(detection.get("confidence", 0.0))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} ({class_id}) conf={confidence:.2f}",
                (x_min, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if log_detections:
            if not out:
                print(f"[F{frame_idx:06d}] detections=0")
            else:
                print(f"[F{frame_idx:06d}] detections={len(out)}")
                for i, detection in enumerate(out):
                    (x1, y1), (x2, y2) = detection["bbox"]
                    class_id = int(detection.get("id", -1))
                    class_name = label_for_class_id(class_id, labels)
                    print(
                        f"  [{i}] label={class_name} id={class_id} "
                        f"conf={float(detection.get('confidence', 0.0)):.3f} "
                        f"bbox=(({int(x1)},{int(y1)}),({int(x2)},{int(y2)}))"
                    )
        if not out:
            cv2.putText(
                frame,
                "NO DETECTIONS",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
