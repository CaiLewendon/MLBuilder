#!/usr/bin/env python3
"""MAVLink-triggered manual photo service (always-on Pi systemd service).

Listens promiscuously on the MAVLink stream for a COMMAND_LONG fired by a
QGroundControl custom action and snaps a still from the requested camera:
  param1 == 1  -> gun_high   (gimbal/gun camera; runs the purple->blue wetness
                              check and advances the shared target counter when
                              confidently WETTED)
  param1 == 2  -> front_high (context shot; no wetness check, no advance)

Photos go to the SAME centralized tree + naming convention as the autonomous
engagement scripts (/images/<camera>/Task_2_<team>_target_<#>_<ts>.jpg) using
the SAME shared counter (/images/target_state.json), so autonomous and manual
captures stay in one extinguishing-order sequence ready for Google-Drive upload.

IMPORTANT — routing: the Pi only SEES a QGC->autopilot COMMAND_LONG if your
mavlink-router mirrors inbound GCS commands to this endpoint. The startup
diagnostic logs every inbound COMMAND_LONG so you can confirm this; if a button
press produces no [CMD] line, enable command forwarding in the router (or fall
back to a relay-bit trigger).

Run:
  python3 -B mavlink_photo_service.py --config /images/photo_service_config.json
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wetness_check  # noqa: E402
import target_photo  # noqa: E402
from target_tracker import TargetTracker  # noqa: E402

from pymavlink import mavutil  # noqa: E402

# MAV_CMD_DO_AUX_FUNCTION (218) — matches this team's existing QGC custom-action
# convention (displacement/radio actions use 218 with param1 as a scripting-AUX
# selector). param1 is the selector the service dispatches on; param2 is the
# switch level (HIGH=2) set by QGC and ignored here.
MAV_CMD_DEFAULT = 218

DEFAULT_CONFIG = {
    "team_name": "unknown",
    "mavlink": "tcp:10.42.0.1:5760",
    # Announce as a component of the vehicle (sys 1) so mavlink-router delivers
    # vehicle-targeted COMMAND_LONGs (the QGC buttons) to this endpoint. Without
    # this, a passive listener only gets broadcast telemetry, never commands.
    "source_system": 1,
    "source_component": 191,
    "announce_heartbeat": True,
    "heartbeat_hz": 1.0,
    "photo_cmd": MAV_CMD_DEFAULT,
    "actions": {
        "304": {"camera": "gun_high", "wetness_check": True, "advance_on_wetted": True},
        "305": {"camera": "front_high", "wetness_check": False, "advance_on_wetted": False},
    },
    "rtsp_base": "rtsp://10.42.0.1:8554",
    "streams": {"gun_high": "gun_high", "front_high": "front_high"},
    "out_root": "/images",
    "state_file": "/images/target_state.json",
    "detect_model": False,
    "model_path": "~/FullDataSetProdV4b_depheavy_edgetpu.tflite",
    "heartbeat_timeout": 30.0,
    "diagnostic_seconds": 20.0,
    "debounce_seconds": 2.0,
    "burst_count": 5,
    "burst_warmup_frames": 8,
    "burst_interval": 0.1,
    "wetness": dict(wetness_check.DEFAULT_CFG),
}


def build_rtsp_pipeline(rtsp_url, latency=200):
    """Single-slot drop-old appsink pipeline. tcp-timeout makes rtspsrc give up
    (instead of hanging forever) if the stream stalls on an on-demand grab."""
    return (
        f"rtspsrc location={rtsp_url} latency={int(latency)} "
        "tcp-timeout=8000000 timeout=8000000 ! "
        "rtpjitterbuffer latency=200 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! "
        "videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path:
        p = Path(path).expanduser()
        if p.is_file():
            user = json.loads(p.read_text())
            for k, v in user.items():
                if k == "wetness" and isinstance(v, dict):
                    cfg["wetness"].update(v)
                else:
                    cfg[k] = v
            print(f"[CONFIG] loaded {p}", flush=True)
        else:
            print(f"[CONFIG] {p} not found — using defaults + writing a template",
                  flush=True)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(cfg, indent=2))
                print(f"[CONFIG] wrote template {p} (edit team_name etc.)", flush=True)
            except Exception as e:
                print(f"[CONFIG] could not write template: {e}", flush=True)
    return cfg


def grab_burst(rtsp_url, count, warmup, interval):
    """Open the RTSP feed on-demand, flush stale frames, grab `count` frames,
    release. Returns a list of BGR frames (possibly empty)."""
    cap = cv2.VideoCapture(build_rtsp_pipeline(rtsp_url), cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames = []
    try:
        if not cap.isOpened():
            print(f"[GRAB] could not open {rtsp_url}", flush=True)
            return frames
        for _ in range(max(0, warmup)):  # flush latency / connect warmup
            cap.read()
            time.sleep(0.02)
        for _ in range(max(1, count)):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(max(0.0, interval))
    finally:
        cap.release()
    return frames


def _sharpness(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


class PhotoService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracker = TargetTracker(
            state_path=cfg["state_file"],
            team_name=(None if cfg["team_name"] == "unknown" else cfg["team_name"]),
        )
        self._last_fire = {}  # debounce: (cmd,param1) -> mono_time
        self._send_lock = threading.Lock()    # serialize MAVLink sends (hb + ack)
        self._capture_lock = threading.Lock()  # one capture at a time
        self._master = None
        self.model = None
        if cfg.get("detect_model"):
            try:
                from MLBuilder.model.tflite.tflitemodel import TFLiteModel
                self.model = TFLiteModel(str(Path(cfg["model_path"]).expanduser()))
                self.model.allocate(tpu=False)  # CPU — avoid Coral contention
                print("[MODEL] loaded for bbox-assisted wetness (CPU)", flush=True)
            except Exception as e:
                print(f"[MODEL] load failed ({e}); falling back to center-crop",
                      flush=True)
                self.model = None

    def _detect_bbox(self, frame):
        if self.model is None:
            return None
        try:
            dets = self.model.detect(frame, nms=True, tol=0.05)
            if not dets:
                return None
            best = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
            return best.get("bbox")
        except Exception:
            return None

    def capture(self, action_key, action):
        camera = action["camera"]
        stream = self.cfg["streams"].get(camera, camera)
        rtsp_url = f"{self.cfg['rtsp_base']}/{stream}"
        frames = grab_burst(rtsp_url, self.cfg["burst_count"],
                            self.cfg["burst_warmup_frames"], self.cfg["burst_interval"])
        if not frames:
            print(f"[CAPTURE] {camera}: no frames grabbed; aborting this shot",
                  flush=True)
            return
        best = max(frames, key=_sharpness)
        target_num = self.tracker.current_target()
        team = self.tracker.get_team()
        # Always save the photo — wetness only gates the counter, never evidence.
        try:
            path = target_photo.save_target_photo(
                best, camera=camera, team=team, target_num=target_num,
                out_root=self.cfg["out_root"])
            print(f"[CAPTURE] {camera} -> {path}  (target #{target_num})", flush=True)
        except Exception as e:
            print(f"[CAPTURE ERROR] save failed: {e}", flush=True)
            return

        if not action.get("wetness_check"):
            print("[CAPTURE] context shot (no wetness check / no advance).", flush=True)
            return

        # Manual path has no pre-fire baseline -> absolute classification.
        wcfg = self.cfg["wetness"]
        results = [wetness_check.assess_absolute(f, self._detect_bbox(f), wcfg)
                   for f in frames]
        state, blue_r, purple_r, npx = wetness_check.aggregate(results)
        print(f"[WETNESS] result={state} blue={blue_r:.2f} purple={purple_r:.2f} "
              f"n_px={npx} (absolute)", flush=True)
        if state == "WETTED" and action.get("advance_on_wetted"):
            used = self.tracker.advance_target(f"manual {camera} wetted")
            print(f"[WETNESS] target #{used} CONFIRMED extinguished; "
                  f"counter -> {self.tracker.current_target()}", flush=True)
        else:
            print(f"[WETNESS] target #{target_num} NOT auto-confirmed ({state}); "
                  f"photo saved, counter NOT advanced — confirm visually.", flush=True)

    def _dispatch(self, cmd, param1):
        key = str(int(round(param1)))
        action = self.cfg["actions"].get(key)
        if action is None:
            print(f"[CMD] cmd={cmd} param1={param1} — no action mapped (keys="
                  f"{list(self.cfg['actions'])})", flush=True)
            return
        now = time.monotonic()
        last = self._last_fire.get((cmd, key), 0.0)
        if now - last < self.cfg["debounce_seconds"]:
            print(f"[CMD] cmd={cmd} param1={key} ignored (debounce)", flush=True)
            return
        self._last_fire[(cmd, key)] = now
        print(f"[CMD] cmd={cmd} param1={key} -> capture {action['camera']}",
              flush=True)
        # Run the capture in a worker thread so a slow/hung RTSP open can NEVER
        # block the MAVLink receive loop (which would drop all later commands).
        threading.Thread(target=self._capture_worker, args=(key, action),
                         daemon=True).start()

    def _capture_worker(self, key, action):
        # Serialize captures; if one hangs it won't stop the rx loop, only the
        # next capture waits. Skip if a capture is already running.
        if not self._capture_lock.acquire(blocking=False):
            print(f"[CAPTURE] busy — skipping {action['camera']} (a capture is "
                  f"already running)", flush=True)
            return
        try:
            self.capture(key, action)
        except Exception as e:
            print(f"[CAPTURE ERROR] {e}", flush=True)
        finally:
            self._capture_lock.release()

    def _start_heartbeat(self, master, cfg):
        """Emit a heartbeat as a vehicle component so mavlink-router routes
        vehicle-targeted commands (QGC buttons) to this endpoint."""
        hz = float(cfg.get("heartbeat_hz", 1.0))
        period = max(0.2, 1.0 / hz) if hz > 0 else 1.0

        def _loop():
            while True:
                try:
                    with self._send_lock:
                        master.mav.heartbeat_send(
                            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                except Exception:
                    pass
                time.sleep(period)

        threading.Thread(target=_loop, daemon=True).start()

    def run(self):
        cfg = self.cfg
        print(f"[MAVLINK] connecting {cfg['mavlink']} ...", flush=True)
        master = mavutil.mavlink_connection(
            cfg["mavlink"],
            source_system=int(cfg.get("source_system", 1)),
            source_component=int(cfg.get("source_component", 191)))
        if cfg.get("announce_heartbeat", True):
            self._start_heartbeat(master, cfg)
            print(f"[MAVLINK] announcing heartbeat as sys={cfg.get('source_system', 1)}"
                  f" comp={cfg.get('source_component', 191)} (router will route "
                  f"vehicle-targeted commands here)", flush=True)
        master.wait_heartbeat(timeout=cfg["heartbeat_timeout"])
        print(f"[MAVLINK] heartbeat from sys={master.target_system} "
              f"comp={master.target_component}", flush=True)
        print(f"[SERVICE] team={self.tracker.get_team()!r} "
              f"next_target={self.tracker.current_target()} "
              f"photo_cmd={cfg['photo_cmd']} actions={list(cfg['actions'])}",
              flush=True)
        print("[SERVICE] Listening for COMMAND_LONG. If a QGC button press shows "
              "no [CMD] line, your mavlink-router is not mirroring inbound "
              "commands to this endpoint — enable forwarding.", flush=True)

        start = time.monotonic()
        seen_any_cmd = False
        diag_window = float(cfg["diagnostic_seconds"])
        diag_hint_done = False
        while True:
            try:
                msg = master.recv_match(blocking=True, timeout=0.5)
            except Exception as e:
                print(f"[MAVLINK] recv error: {e}; reconnecting in 1s", flush=True)
                time.sleep(1.0)
                continue
            now = time.monotonic()
            if (not diag_hint_done and not seen_any_cmd
                    and now - start > diag_window):
                print(f"[DIAG] no COMMAND_LONG seen in {diag_window:.0f}s. Telemetry "
                      "is flowing but inbound commands are not — check router "
                      "forwarding (or switch to a relay-bit trigger).", flush=True)
                diag_hint_done = True
            if msg is None:
                continue
            t = msg.get_type()
            if t in ("COMMAND_LONG", "COMMAND_INT"):
                seen_any_cmd = True
                cmd = int(getattr(msg, "command", -1))
                p1 = float(getattr(msg, "param1", 0.0))
                if cmd == int(cfg["photo_cmd"]):
                    # ACK so QGC doesn't show "vehicle did not respond". Sent
                    # immediately (before the multi-second capture). Use the
                    # 2-arg broadcast form — universally supported across
                    # pymavlink dialects; mavp2p broadcasts it to QGC.
                    try:
                        with self._send_lock:
                            master.mav.command_ack_send(cmd, 0)  # 0 = ACCEPTED
                    except Exception as e:
                        print(f"[ACK] command_ack_send failed: {e}", flush=True)
                    self._dispatch(cmd, p1)
                else:
                    # Diagnostic: surface any inbound command so the operator can
                    # read off what their QGC button actually sends.
                    print(f"[CMD?] inbound {t} cmd={cmd} param1={p1} "
                          f"(not photo_cmd={cfg['photo_cmd']})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="MAVLink-triggered manual photo service")
    ap.add_argument("--config", type=str, default="/images/photo_service_config.json",
                    help="Path to the JSON config (action map, streams, wetness, etc.). "
                         "A template is written here if absent.")
    ap.add_argument("--mavlink", type=str, default=None,
                    help="Override the MAVLink endpoint from config.")
    ap.add_argument("--reset-counter", action="store_true",
                    help="Reset the shared target counter to 1 and exit (run "
                         "once at boot so target numbers restart each power cycle).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mavlink:
        cfg["mavlink"] = args.mavlink
    if args.reset_counter:
        TargetTracker(state_path=cfg["state_file"], team_name=cfg["team_name"],
                      start_target=1)
        print(f"[RESET] target counter -> 1 at {cfg['state_file']} "
              f"(team={cfg['team_name']})", flush=True)
        return
    PhotoService(cfg).run()


if __name__ == "__main__":
    main()
