import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_video_source(video_source: str) -> int | str:
    try:
        return int(video_source)
    except ValueError:
        return video_source


def load_labels_from_txt(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [label for label in labels if label]


def load_labels_from_metadata(path: Path) -> list[str]:
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed. Skipping metadata labels.")
        return []

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, dict):
        labels: list[str] = []
        for idx in sorted(names.keys(), key=lambda x: int(x)):
            labels.append(str(names[idx]))
        return labels
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def load_labels(model_path: Path, labels_path: str | None, metadata_path: str | None) -> list[str]:
    if labels_path:
        labels_file = Path(labels_path)
        if not labels_file.is_file():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        return load_labels_from_txt(labels_file)

    if metadata_path:
        metadata_file = Path(metadata_path)
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        return load_labels_from_metadata(metadata_file)

    auto_metadata = model_path.with_name("metadata.yaml")
    if auto_metadata.is_file():
        return load_labels_from_metadata(auto_metadata)
    return []


def resolve_label(class_id: int, labels: list[str]) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


def bbox_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    denom = a1 + a2 - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def dominant_color_name(mean_hsv: list[float]) -> str:
    h, s, v = mean_hsv
    if v < 40:
        return "black"
    if s < 30 and v > 200:
        return "white"
    if s < 30:
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 135:
        return "blue"
    if h < 160:
        return "purple"
    return "pink"


def extract_color_features(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> dict | None:
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    mean_bgr = roi.reshape(-1, 3).mean(axis=0).tolist()
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_hsv = roi_hsv.reshape(-1, 3).mean(axis=0).tolist()
    return {
        "mean_bgr": [round(float(c), 2) for c in mean_bgr],
        "mean_hsv": [round(float(c), 2) for c in mean_hsv],
        "dominant_color": dominant_color_name(mean_hsv),
    }


def attach_tracking_status(
    detections: list[dict],
    previous_tracks: list[dict],
    next_track_id: int,
) -> tuple[list[dict], list[dict], list[int], int]:
    used_prev: set[int] = set()
    current_tracks: list[dict] = []

    for det in detections:
        bbox = (
            det["bbox_top_left"][0],
            det["bbox_top_left"][1],
            det["bbox_bottom_right"][0],
            det["bbox_bottom_right"][1],
        )
        class_id = det["class_id"]

        best_idx = -1
        best_iou = 0.0
        for idx, prev in enumerate(previous_tracks):
            if idx in used_prev:
                continue
            if prev["class_id"] != class_id:
                continue
            iou = bbox_iou(bbox, prev["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx != -1 and best_iou >= 0.30:
            used_prev.add(best_idx)
            track_id = previous_tracks[best_idx]["track_id"]
            status = "stable" if best_iou >= 0.70 else "changed"
        else:
            track_id = next_track_id
            next_track_id += 1
            status = "new"

        det["track_id"] = track_id
        det["status"] = status
        current_tracks.append({"track_id": track_id, "class_id": class_id, "bbox": bbox})

    lost_track_ids = [
        prev["track_id"] for idx, prev in enumerate(previous_tracks) if idx not in used_prev
    ]
    return detections, current_tracks, lost_track_ids, next_track_id


def main():
    parser = argparse.ArgumentParser(
        prog="tf_inference_data_live",
        description="Live TFLite inference with structured per-frame data output",
    )
    parser.add_argument("model", type=str, help="Path to .tflite model")
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="0",
        help="Video source (camera index, file path, or stream URL)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--runtime-nms",
        "-n",
        action="store_true",
        help="Apply runtime NMS (use if model was exported without built-in NMS)",
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=None,
        help="Optional labels text file (one class name per line)",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        default=None,
        help="Optional metadata.yaml with names mapping",
    )
    parser.add_argument(
        "--source-id",
        "-s",
        type=str,
        default=None,
        help="Optional source id to include in output payload",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable OpenCV display window (prints JSON only)",
    )

    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    try:
        labels = load_labels(model_path, args.labels, args.metadata)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if labels:
        print(f"Loaded {len(labels)} labels")
    else:
        print("No labels loaded; using class_<id> fallback names")

    source_value = parse_video_source(args.video)
    source_id = args.source_id or str(args.video)
    cap = cv2.VideoCapture(source_value)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.video}")
        sys.exit(1)

    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    model = TFLiteModel(str(model_path))
    model.allocate()

    frame_id = 0
    previous_tracks: list[dict] = []
    next_track_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame")
            continue

        frame_id += 1
        now = time.time()
        raw_detections = model.detect(frame, nms=args.runtime_nms, tol=args.confidence)

        detections: list[dict] = []
        frame_h, frame_w = frame.shape[:2]
        for det in raw_detections:
            x1, y1 = int(det["bbox"][0][0]), int(det["bbox"][0][1])
            x2, y2 = int(det["bbox"][1][0]), int(det["bbox"][1][1])
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(det.get("id", -1))
            class_name = resolve_label(class_id, labels)
            confidence = float(det.get("confidence", 0.0))
            width = x2 - x1
            height = y2 - y1
            area = width * height
            color_features = extract_color_features(frame, (x1, y1, x2, y2))

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox_top_left": [x1, y1],
                    "bbox_bottom_right": [x2, y2],
                    "width": width,
                    "height": height,
                    "area": area,
                    "color_features": color_features,
                }
            )

        detections, current_tracks, lost_tracks, next_track_id = attach_tracking_status(
            detections, previous_tracks, next_track_id
        )
        previous_tracks = current_tracks

        payload = {
            "timestamp": now,
            "frame_id": frame_id,
            "source_id": source_id,
            "detections": detections,
        }
        if lost_tracks:
            payload["lost_track_ids"] = lost_tracks

        if detections or lost_tracks:
            print(json.dumps(payload, separators=(",", ":")))

        if not args.no_window:
            for det in detections:
                x1, y1 = det["bbox_top_left"]
                x2, y2 = det["bbox_bottom_right"]
                label_text = (
                    f"{det['class_name']} {det['confidence']:.2f} "
                    f"T{det['track_id']} {det['status']}"
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label_text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Inference Data Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
