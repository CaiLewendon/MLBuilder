#!/usr/bin/env python3
import shlex
import signal
import subprocess
import sys
import time
import cv2

# HOST = "100.80.163.116"
HOST = "caile-predator-pt315-53"
PORT = 5600

WIDTH = 640
HEIGHT = 480
FPS = 10
QUANTIZER = 16

# def build_gstreamer_command() -> list[str]:
pipeline1 = (
    # 'gst-launch-1.0 -e '
    'libcamerasrc camera-name="/base/axi/pcie@1000120000/rp1/i2c@80000/imx219@10" ! '
    f'video/x-raw,width={WIDTH},height={HEIGHT},format=BGR,framerate={FPS}/1 ! '
    'appsink drop=true max-buffers=1 sync=false'
)
pipeline2 = (
    'appsrc !'
    'queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! '
    'videoconvert ! '
    'video/x-raw,format=I420 ! '
    'x264enc '
    'speed-preset=medium '
    'tune=zerolatency '
    f'pass=qual quantizer={QUANTIZER} '
    'key-int-max=60 '
    'bframes=0 '
    'rc-lookahead=0 '
    'byte-stream=true '
    'threads=4 '
    'sliced-threads=true ! '
    'video/x-h264,profile=main ! '
    'h264parse config-interval=-1 ! '
    'rtph264pay pt=96 config-interval=1 mtu=1000 aggregate-mode=none ! '
    f'udpsink host={HOST} port={PORT} sync=false async=false'
)
    # return shlex.split(pipeline)

def main():
    # cmd = build_gstreamer_command()

    cap = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    writer = cv2.VideoWriter(pipeline2, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)

    if not writer.isOpened():
        raise RuntimeError("Failed to open GStreamer VideoWriter")

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    print(f"Streaming to {HOST}:{PORT}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"FPS: {FPS}")
    print(f"Quantizer: {QUANTIZER}")
    print("Press Ctrl+C to stop.\n")
    # print("Command:")
    # print(" ".join(cmd))
    print()

    # try:
    #     proc = subprocess.Popen(cmd)
    # except FileNotFoundError:
    #     print("Error: gst-launch-1.0 not found.", file=sys.stderr)
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"Error starting pipeline: {e}", file=sys.stderr)
    #     sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()

            writer.write(frame.astype("uint8"))

            # ret = proc.poll()
            # if ret is not None:
            #     print(f"GStreamer exited with code {ret}", file=sys.stderr)
            #     sys.exit(ret)
            # time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    #     proc.send_signal(signal.SIGINT)

    #     try:
    #         proc.wait(timeout=5)
    #     except subprocess.TimeoutExpired:
    #         proc.terminate()
    #         try:
    #             proc.wait(timeout=3)
    #         except subprocess.TimeoutExpired:
    #             proc.kill()

    # sys.exit(proc.returncode if proc.returncode is not None else 0)

if __name__ == "__main__":
    main()
