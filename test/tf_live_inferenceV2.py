import sys
import time
from pathlib import Path
import threading
import cv2
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = 10
WIDTH = 640
HEIGHT = 360

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
    "rtspsrc location=rtsp://10.42.0.1:8554/front_high latency=0 !"
    "rtph264depay !"
    "h264parse !"
    "avdec_h264 !"
    "videoconvert !"
    "appsink drop=true max-buffers=1 sync=false"  # drop stale frames, don't block
)

pipeline4 = (
    "appsrc ! "
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


def main():
    parser = argparse.ArgumentParser(prog="tflive", description="TF Live Inference Model Test")
    parser.add_argument("model", nargs="?", help="TFlite model to run", type=str)
    parser.add_argument("--nms", "-n", action="store_false")
    parser.add_argument("--tpu", "-t", action="store_true")
    parser.add_argument("--confidence", "-c", type=float, default=0.25)
    parser.add_argument("--labels", "-l", type=str, default=None)
    parser.add_argument("--process", "-p", action="store_true", help="Enable inference processing")
    parser.add_argument("--overlay", "-o", action="store_true", help="Draw detection labels/boxes on output video")
    args = parser.parse_args()

    if args.process and not args.model:
        parser.error("--process requires a model argument")

    cap = cv2.VideoCapture(pipeline3, cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    writer = cv2.VideoWriter(pipeline4, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)

    if not writer.isOpened():
        raise RuntimeError("Failed to open GStreamer VideoWriter")
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    # -------------------------------------------------------------------------
    # PROCESSING MODE: inference + optional detection overlay
    # -------------------------------------------------------------------------
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
        frame_event = threading.Event()  # signals inference thread that a new frame is ready

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
                frame_event.set()  # wake inference thread

        def inference_thread():
            count = 0
            while not stop_event.is_set():
                # block until a new frame arrives, timeout so we can check stop_event
                frame_event.wait(timeout=0.1)
                frame_event.clear()
                with frame_lock:
                    frame = latest_frame[0]
                if frame is None:
                    continue
                detections = m.detect(frame, nms=args.nms, tol=args.confidence)
                count+=1
                if count % 30 == 0:
                    print(f"infer_frames={count} dets={len(detections)}")
                if detections:
                    print(f"detections={len(detections)} first={detections[0]}")
                with det_lock:
                    latest_detections[0] = detections

        t_cap = threading.Thread(target=capture_thread, daemon=True)
        t_inf = threading.Thread(target=inference_thread, daemon=True)
        t_cap.start()
        t_inf.start()

        frame_interval = 1.0 / FPS

        try:
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

                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = frame.copy()

                writer.write(frame)

                # pace the main loop to FPS, yield remaining time to OS
                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stop_event.set()
            cap.release()
            writer.release()

    # -------------------------------------------------------------------------
    # PASSTHROUGH MODE: just forward video, no inference
    # -------------------------------------------------------------------------
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
                writer.write(frame)

                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            writer.release()


if __name__ == "__main__":
    main()