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

    if crop_pass and not (0.0 < crop_ratio <= 1.0):
        parser.error("--center-crop-ratio must be > 0.0 and <= 1.0")

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    max_skips = 5
    frame_skip = 0

    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    m = TFLiteModel(model_path)
    m.allocate(tpu=use_tpu)

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

        out = m.detect(frame, nms=use_nms, tol=tolerance)
        if crop_pass:
            cropped, x_off, y_off = center_crop(frame, crop_ratio)
            crop_out = m.detect(cropped, nms=use_nms, tol=tolerance)
            out.extend(remap_detections(crop_out, x_off, y_off))

        for detection in out:
            bbox = detection["bbox"]
            x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
            x_max, y_max = int(bbox[1][0]), int(bbox[1][1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
