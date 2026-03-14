import sys
from pathlib import Path

import cv2
import argparse
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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

    args = parser.parse_args()
    try:
        video_source = int(args.video)
    except:
        video_source = args.video
    model_path = args.model
    use_nms = args.nms
    use_tpu = args.tpu
    tolerance = args.confidence

    pipeline = (
        'libcamerasrc camera-name="/base/axi/pcie@1000120000/rp1/i2c@80000/imx219@10" ! video/x-raw,width=640,height=480,format=NV12 ! videoconvert ! appsink'  
    )

    #cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture('rtsp://root:12345@10.42.0.2/stream=0')

    gst_out = (
    f'appsrc ! '
    f'videoconvert ! '
    f'video/x-raw,format=I420,width=640,height=480,framerate=30/1 ! '
    f'x264enc ! '
    f'rtph264pay config-interval=1 pt=96 ! '
    #f'jpegenc quality=85 ! '
    #f'rtpjpegpay ! '
    #f'udpsink host=100.80.163.116 port=5600'
    f'udpsink host=100.73.135.84 port=5600'
)

    writer = cv2.VideoWriter(
        gst_out,
        cv2.CAP_GSTREAMER,
        0,
        30,
        (640, 480),
        True
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to open GStreamer VideoWriter")

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

        print(json.dumps(out, indent=2))

        for detection in out:
            bbox = detection["bbox"]
            x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
            x_max, y_max = int(bbox[1][0]), int(bbox[1][1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        writer.write(frame)


if __name__ == "__main__":
    main()
