import gi, time
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
p = Gst.parse_launch('rtspsrc location=rtsp://10.42.0.1:8554/front_high latency=200 ! rtpjitterbuffer latency=200 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink name=s emit-signals=true sync=false drop=false max-buffers=1')
sink = p.get_by_name('s')
n = [0]
def on_sample(s):
    n[0] += 1
    s.emit('pull-sample')
    return 0
sink.connect('new-sample', on_sample)
p.set_state(Gst.State.PLAYING)
t0 = time.monotonic()
try:
    while time.monotonic() - t0 < 10:
        time.sleep(1)
        e = time.monotonic() - t0
        print(f'{e:4.1f}s: frames={n[0]:4d}  fps={n[0]/e:5.2f}')
finally:
    p.set_state(Gst.State.NULL)