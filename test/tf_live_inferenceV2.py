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


def main():
    parser = argparse.ArgumentParser(prog="tflive", description="TF Live Inference Model Test")
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
        exit()

    if args.process:
        try:
            labels = load_labels(args.labels)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        from MLBuilder.model.tflite.tflitemodel import TFLiteModel
        m = TFLiteModel(args.model)
        m.allocate(tpu=args.tpu)

        latest_frame = [None]
        latest_detections = [[]]
        frame_lock = threading.Lock()
        det_lock = threading.Lock()
        stop_event = threading.Event()
        frame_event = threading.Event()

        frame_seq = [0]

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
                if detections:
                    print(f"detections={len(detections)} first={detections[0]} count={count} time={time.monotonic()}", flush=True)

                with det_lock:
                    latest_detections[0] = detections

                if detections:
                    print(f"[WRITE] seq={frame_seq[0]} dets={len(detections)} first={detections[0]}", flush=True)

        t_cap = threading.Thread(target=capture_thread, daemon=True)
        t_inf = threading.Thread(target=inference_thread, daemon=True)
        t_cap.start()
        t_inf.start()

        frame_interval = 1.0 / FPS

        try:
            if args.no_output:
                # Capture + inference only — keep main thread alive, no writer work
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

                    if args.overlay:
                        for detection in detections:
                            bbox = detection["bbox"]
                            x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
                            x_max, y_max = int(bbox[1][0]), int(bbox[1][1])
                            class_id = int(detection.get("id", -1))
                            confidence = float(detection.get("confidence", 0.0))
                            label = resolve_label(class_id, labels)

                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            text = f"{label} {confidence:.2f}"
                            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (x_min, y_min - th - 6), (x_min + tw, y_min), (0, 255, 0), -1)
                            cv2.putText(frame, text, (x_min, y_min - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

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
            print("\nStopping...")
        finally:
            stop_event.set()
            t_cap.join(timeout=2.0)
            t_inf.join(timeout=2.0)
            cap.release()
            if writer is not None:
                writer.release()

    else:
        frame_interval = 1.0 / FPS
        try:
            while True:
                loop_start = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.01)
                    continue
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = frame.copy()
                if writer is not None:
                    writer.write(frame)

                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            if writer is not None:
                writer.release()


if __name__ == "__main__":
    main()
