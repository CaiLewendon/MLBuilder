"""
Manual gimbal control + live telemetry monitor.

Continuous keyboard control of gimbal pitch/yaw using MAV_CMD_DO_MOUNT_CONTROL
(same command shape as tf_live_infrence_gimbal_live.py), with a refreshing
telemetry panel that confirms the autopilot link is alive by displaying the
last value and age of HEARTBEAT, ATTITUDE, SYS_STATUS, GLOBAL_POSITION_INT,
VFR_HUD, GPS_RAW_INT, MOUNT_STATUS, GIMBAL_DEVICE_ATTITUDE_STATUS, RC_CHANNELS,
and COMMAND_ACK.

Two run modes:

  Interactive panel (default — no --live-fire):
    Live gimbal control + telemetry panel. 'f' triggers a BLANK fire (logs only,
    no relay command sent) for workflow testing.

  One-shot fire (--live-fire):
    Connect to MAVLink, send a single MAV_CMD_DO_REPEAT_RELAY, wait briefly for
    the ACK, then exit. No panel, no key handler, no concurrent threads —
    bypasses the interactive loop entirely so the relay command is issued from
    a single linear code path. Parallel design to --center-gimbal in
    tf_live_inferenceV2_gimbal_auto.py.

Interactive panel keys (single keypress, no Enter required; hold to repeat):
  w / s   pitch up / down by step
  a / d   yaw  left / right by step
  c       center (pitch=0, yaw=0)
  + / -   increase / decrease step size
  r       resend current setpoint
  f       simulate FIRE in BLANK mode (logs only)
  F       emergency relay OFF (force fire=off)
  q       quit
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import time
from typing import Optional

GIMBAL_YAW_MIN_DEG = -90.0
GIMBAL_YAW_MAX_DEG = 90.0
GIMBAL_PITCH_MIN_DEG = -45.0
GIMBAL_PITCH_MAX_DEG = 45.0
MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_MOUNT_MODE_MAVLINK_TARGETING = 2
MAV_CMD_DO_SET_RELAY = 181
MAV_CMD_DO_REPEAT_RELAY = 182

STEP_MIN = 0.5
STEP_MAX = 30.0

ANSI_CLEAR_SCREEN = "\x1b[2J"
ANSI_HOME = "\x1b[H"
ANSI_CLEAR_TO_EOL = "\x1b[K"
ANSI_CLEAR_TO_EOS = "\x1b[J"
ANSI_HIDE_CURSOR = "\x1b[?25l"
ANSI_SHOW_CURSOR = "\x1b[?25h"

TRACKED_MESSAGES = (
    "HEARTBEAT",
    "ATTITUDE",
    "SYS_STATUS",
    "GLOBAL_POSITION_INT",
    "VFR_HUD",
    "GPS_RAW_INT",
    "MOUNT_STATUS",
    "GIMBAL_DEVICE_ATTITUDE_STATUS",
    "RC_CHANNELS",
    "COMMAND_ACK",
    "STATUSTEXT",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def send_mount_control(master, pitch_deg: float, yaw_deg: float) -> None:
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_DO_MOUNT_CONTROL,
        0,
        float(pitch_deg),
        0.0,
        float(yaw_deg),
        0.0,
        0.0,
        0.0,
        float(MAV_MOUNT_MODE_MAVLINK_TARGETING),
    )


def send_relay(master, relay_num: int, state: int) -> None:
    """MAV_CMD_DO_SET_RELAY (181). state: 1=on, 0=off. Used here only for SAFETY
    OFF — we forcibly drive the relay low on exit/emergency."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_DO_SET_RELAY,
        0,
        float(int(relay_num)),
        float(int(bool(state))),
        0.0, 0.0, 0.0, 0.0, 0.0,
    )


def send_repeat_relay(master, relay_num: int, cycles: int, cycle_time_s: float) -> None:
    """MAV_CMD_DO_REPEAT_RELAY (182). Mirrors QGroundControl's "Shoot Gun" action:
    param1=relay, param2=cycles, param3=cycle_time. ArduPilot pulses the relay
    for `cycles` cycles, each cycle period `cycle_time` seconds."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_DO_REPEAT_RELAY,
        0,
        float(int(relay_num)),
        float(int(cycles)),
        float(cycle_time_s),
        0.0, 0.0, 0.0, 0.0,
    )


def request_streams(master, rate_hz: int) -> None:
    from pymavlink import mavutil

    try:
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            int(rate_hz),
            1,
        )
    except Exception as e:
        print(f"[MAVLINK] request_data_stream failed: {e}")


class Telemetry:
    def __init__(self) -> None:
        self.last_seen = {}
        self.data = {}
        self.total_msgs = 0
        self.last_send_err: Optional[str] = None
        self.last_send_time: Optional[float] = None
        self.send_count = 0

    def ingest(self, msg) -> None:
        name = msg.get_type()
        self.total_msgs += 1
        if name in TRACKED_MESSAGES:
            self.last_seen[name] = time.monotonic()
            try:
                d = msg.to_dict()
            except Exception:
                d = {}
            d.pop("mavpackettype", None)
            self.data[name] = d

    def age(self, name: str) -> Optional[float]:
        t = self.last_seen.get(name)
        if t is None:
            return None
        return time.monotonic() - t

    def get(self, name: str) -> dict:
        return self.data.get(name, {})


def fmt_age(age: Optional[float]) -> str:
    if age is None:
        return "   --   "
    if age >= 99.0:
        return ">  99.0s"
    return f"{age:6.2f}s "


def fresh_marker(age: Optional[float], threshold: float = 2.0) -> str:
    if age is None:
        return "X"
    if age <= threshold:
        return "+"
    return "!"


class RawTerminal:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old = None
        self.enabled = False

    def __enter__(self):
        try:
            import termios
            import tty
        except ImportError:
            return self
        try:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        except Exception:
            self.old = None
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self) -> Optional[str]:
        if not self.enabled:
            return None
        rlist, _, _ = select.select([self.fd], [], [], 0)
        if not rlist:
            return None
        try:
            ch = os.read(self.fd, 1)
        except (BlockingIOError, OSError):
            return None
        if not ch:
            return None
        try:
            return ch.decode("utf-8", errors="replace")
        except Exception:
            return None


def render(tel: Telemetry, pitch: float, yaw: float, step: float,
           dry_run: bool, master_ok: bool, mav_string: str,
           firing: bool = False, fire_relay: int = 2,
           live_fire: bool = False, fire_elapsed: float = 0.0) -> None:
    out = [ANSI_HOME]

    def line(text: str) -> None:
        out.append(text + ANSI_CLEAR_TO_EOL + "\n")

    tx_state = "DRY-RUN" if dry_run else ("LIVE" if master_ok else "OFFLINE")
    fire_mode_label = "LIVE-FIRE" if live_fire else "BLANK"
    line("=== Manual Gimbal Control + Live Telemetry ===")
    line(f"link: {mav_string}    tx: {tx_state}    msgs_rx: {tel.total_msgs}    fire_mode: {fire_mode_label}")
    line("controls: [w/s] pitch  [a/d] yaw  [c] center  [+/-] step  [r] resend  [f] FIRE  [F] off  [q] quit")
    line("")

    line(f"setpoint  pitch={pitch:+7.2f} deg   yaw={yaw:+7.2f} deg   step={step:5.2f} deg")
    if tel.last_send_time is not None:
        send_age = time.monotonic() - tel.last_send_time
        line(f"          last_tx={send_age:5.2f}s ago   tx_count={tel.send_count}"
             + (f"   ERR: {tel.last_send_err}" if tel.last_send_err else ""))
    else:
        line("          last_tx=  --     tx_count=0")

    if firing:
        line(f">>> PULSING ({fire_mode_label}) relay={fire_relay}  t={fire_elapsed:.2f}s  "
             f"(autopilot owns timing; press F for emergency off) <<<")
    else:
        line(f"fire      idle  relay={fire_relay}  (press f to {'FIRE one burst' if live_fire else 'simulate burst'} via DO_REPEAT_RELAY)")
    line("")
    line("--- live telemetry (+ fresh <2s, ! stale, X never seen) ---")

    hb = tel.get("HEARTBEAT")
    line(f" {fresh_marker(tel.age('HEARTBEAT'))} HEARTBEAT       {fmt_age(tel.age('HEARTBEAT'))}  "
         f"sys_status={hb.get('system_status','?')} "
         f"base_mode={hb.get('base_mode','?')} "
         f"custom_mode={hb.get('custom_mode','?')} "
         f"type={hb.get('type','?')} autopilot={hb.get('autopilot','?')}")

    att = tel.get("ATTITUDE")
    if att:
        line(f" {fresh_marker(tel.age('ATTITUDE'))} ATTITUDE        {fmt_age(tel.age('ATTITUDE'))}  "
             f"roll={math.degrees(att.get('roll',0.0)):+7.2f}d  "
             f"pitch={math.degrees(att.get('pitch',0.0)):+7.2f}d  "
             f"yaw={math.degrees(att.get('yaw',0.0)):+7.2f}d")
    else:
        line(f" {fresh_marker(None)} ATTITUDE        {fmt_age(None)}")

    sys_s = tel.get("SYS_STATUS")
    if sys_s:
        v_mv = sys_s.get("voltage_battery", 0) or 0
        cur_ca = sys_s.get("current_battery", 0) or 0
        rem = sys_s.get("battery_remaining", 0)
        line(f" {fresh_marker(tel.age('SYS_STATUS'))} SYS_STATUS      {fmt_age(tel.age('SYS_STATUS'))}  "
             f"V={v_mv/1000.0:5.2f}  I={cur_ca/100.0:6.2f}A  rem={rem}%")
    else:
        line(f" {fresh_marker(None)} SYS_STATUS      {fmt_age(None)}")

    gp = tel.get("GLOBAL_POSITION_INT")
    if gp:
        line(f" {fresh_marker(tel.age('GLOBAL_POSITION_INT'))} GLOBAL_POSITION {fmt_age(tel.age('GLOBAL_POSITION_INT'))}  "
             f"lat={gp.get('lat',0)/1e7:11.6f}  lon={gp.get('lon',0)/1e7:12.6f}  "
             f"alt={gp.get('alt',0)/1000.0:7.1f}m  relalt={gp.get('relative_alt',0)/1000.0:6.1f}m  "
             f"hdg={gp.get('hdg',0)/100.0:5.1f}d")
    else:
        line(f" {fresh_marker(None)} GLOBAL_POSITION {fmt_age(None)}")

    vfr = tel.get("VFR_HUD")
    if vfr:
        line(f" {fresh_marker(tel.age('VFR_HUD'))} VFR_HUD         {fmt_age(tel.age('VFR_HUD'))}  "
             f"as={vfr.get('airspeed',0.0):4.1f}m/s  gs={vfr.get('groundspeed',0.0):4.1f}m/s  "
             f"alt={vfr.get('alt',0.0):7.1f}m  climb={vfr.get('climb',0.0):+5.1f}m/s  "
             f"thr={vfr.get('throttle',0):3d}%  hdg={vfr.get('heading',0):3d}d")
    else:
        line(f" {fresh_marker(None)} VFR_HUD         {fmt_age(None)}")

    gps = tel.get("GPS_RAW_INT")
    if gps:
        line(f" {fresh_marker(tel.age('GPS_RAW_INT'))} GPS_RAW_INT     {fmt_age(tel.age('GPS_RAW_INT'))}  "
             f"fix={gps.get('fix_type',0)}  sats={gps.get('satellites_visible',0):2d}  "
             f"eph={gps.get('eph',0)}  epv={gps.get('epv',0)}")
    else:
        line(f" {fresh_marker(None)} GPS_RAW_INT     {fmt_age(None)}")

    mnt = tel.get("MOUNT_STATUS")
    if mnt:
        line(f" {fresh_marker(tel.age('MOUNT_STATUS'))} MOUNT_STATUS    {fmt_age(tel.age('MOUNT_STATUS'))}  "
             f"pitch={mnt.get('pointing_a',0)/100.0:+7.2f}d  "
             f"roll={mnt.get('pointing_b',0)/100.0:+7.2f}d  "
             f"yaw={mnt.get('pointing_c',0)/100.0:+7.2f}d")
    else:
        line(f" {fresh_marker(None)} MOUNT_STATUS    {fmt_age(None)}   (autopilot feedback for gimbal angle)")

    gda = tel.get("GIMBAL_DEVICE_ATTITUDE_STATUS")
    if gda:
        q = gda.get("q") or [1.0, 0.0, 0.0, 0.0]
        try:
            qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        except Exception:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        roll = math.degrees(math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
        pitch_g = math.degrees(math.asin(clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
        yaw_g = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
        line(f" {fresh_marker(tel.age('GIMBAL_DEVICE_ATTITUDE_STATUS'))} GIMBAL_DEV_ATT  {fmt_age(tel.age('GIMBAL_DEVICE_ATTITUDE_STATUS'))}  "
             f"roll={roll:+7.2f}d  pitch={pitch_g:+7.2f}d  yaw={yaw_g:+7.2f}d  flags={gda.get('flags','?')}")
    else:
        line(f" {fresh_marker(None)} GIMBAL_DEV_ATT  {fmt_age(None)}   (MAVLink v2 gimbal protocol)")

    rc = tel.get("RC_CHANNELS")
    if rc:
        chans = [rc.get(f"chan{i}_raw", 0) for i in range(1, 9)]
        line(f" {fresh_marker(tel.age('RC_CHANNELS'))} RC_CHANNELS     {fmt_age(tel.age('RC_CHANNELS'))}  "
             f"1-8={chans}  rssi={rc.get('rssi','?')}")
    else:
        line(f" {fresh_marker(None)} RC_CHANNELS     {fmt_age(None)}")

    ack = tel.get("COMMAND_ACK")
    if ack:
        line(f" {fresh_marker(tel.age('COMMAND_ACK'))} COMMAND_ACK     {fmt_age(tel.age('COMMAND_ACK'))}  "
             f"cmd={ack.get('command','?')}  result={ack.get('result','?')}  "
             f"progress={ack.get('progress','?')}")
    else:
        line(f" {fresh_marker(None)} COMMAND_ACK     {fmt_age(None)}   (waiting for first cmd ack)")

    st = tel.get("STATUSTEXT")
    if st:
        text = (st.get("text") or "").strip()
        line(f" {fresh_marker(tel.age('STATUSTEXT'))} STATUSTEXT      {fmt_age(tel.age('STATUSTEXT'))}  "
             f"sev={st.get('severity','?')}  {text[:80]}")
    else:
        line(f" {fresh_marker(None)} STATUSTEXT      {fmt_age(None)}")

    out.append(ANSI_CLEAR_TO_EOS)
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="manual_gimbal_control",
        description="Continuous keyboard gimbal control with live MAVLink telemetry panel.",
    )
    parser.add_argument("--mavlink", type=str, default="tcp:10.42.0.1:5760",
                        help="pymavlink connection string (default: tcp:10.42.0.1:5760)")
    parser.add_argument("--no-mavlink", action="store_true",
                        help="Skip MAVLink connect; print-only dry-run.")
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0,
                        help="Seconds to wait for first heartbeat (default: 15).")
    parser.add_argument("--stream-rate", type=int, default=10,
                        help="Requested telemetry stream rate from autopilot, Hz (default: 10).")
    parser.add_argument("--send-rate", type=float, default=20.0,
                        help="Max gimbal command send rate, Hz (default: 20).")
    parser.add_argument("--heartbeat-send-rate", type=float, default=2.0,
                        help="Background resend rate for the current setpoint, Hz (default: 2).")
    parser.add_argument("--redraw-rate", type=float, default=10.0,
                        help="Telemetry panel refresh rate, Hz (default: 10).")
    parser.add_argument("--step", type=float, default=2.0,
                        help="Degrees per keypress (default: 2.0).")
    parser.add_argument("--initial-pitch", type=float, default=0.0)
    parser.add_argument("--initial-yaw", type=float, default=0.0)
    parser.add_argument("--fire-relay", type=int, default=1,
                        help="Relay instance for fire trigger (param1 of MAV_CMD_DO_REPEAT_RELAY). "
                             "Matches the QGroundControl 'Shoot Gun' action default (relay 1).")
    parser.add_argument("--fire-cycles", type=int, default=1,
                        help="Cycles per fire press (param2 of MAV_CMD_DO_REPEAT_RELAY). Default 1.")
    parser.add_argument("--fire-period", type=float, default=2.0,
                        help="Cycle period in seconds (param3 of MAV_CMD_DO_REPEAT_RELAY). Default 2.0.")
    parser.add_argument("--live-fire", action="store_true",
                        help="DANGEROUS. One-shot fire-and-exit. With this flag set the script "
                             "connects to MAVLink, sends a single MAV_CMD_DO_REPEAT_RELAY "
                             "(matching the QGroundControl 'Shoot Gun' action), waits up to ~2.5s "
                             "for the COMMAND_ACK, then exits. No panel, no key loop. Without this "
                             "flag the script opens the interactive panel for gimbal control + "
                             "BLANK fire simulation.")
    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step must be > 0")
    if args.send_rate <= 0 or args.redraw_rate <= 0:
        parser.error("--send-rate and --redraw-rate must be > 0")

    # --live-fire: one-shot fire-and-exit, parallel to --center-gimbal in the
    # autonomous script. Bypasses the interactive panel and key handler entirely
    # so the relay command is issued from a single linear code path with no
    # repaint loop or repeat-key dispatch in the picture.
    if args.live_fire:
        if args.no_mavlink:
            print("[FIRE] --live-fire requires a MAVLink connection (conflict with --no-mavlink)")
            return 2
        try:
            from pymavlink import mavutil
        except ImportError:
            print("[MAVLINK] pymavlink not installed.")
            return 2
        print(f"[MAVLINK] Connecting: {args.mavlink}")
        try:
            master = mavutil.mavlink_connection(args.mavlink)
            master.wait_heartbeat(timeout=args.heartbeat_timeout)
            print(f"[MAVLINK] heartbeat sysid={master.target_system} "
                  f"compid={master.target_component}")
        except Exception as e:
            print(f"[MAVLINK] connect failed: {e}")
            return 2
        print(f"[FIRE ONE-SHOT] sending DO_REPEAT_RELAY(182) "
              f"param1={args.fire_relay} param2={args.fire_cycles} "
              f"param3={args.fire_period:.2f}s")
        try:
            send_repeat_relay(master, args.fire_relay, args.fire_cycles, args.fire_period)
        except Exception as e:
            print(f"[FIRE] send failed: {e}")
            try:
                master.close()
            except Exception:
                pass
            return 2
        pulse_duration = float(args.fire_cycles) * float(args.fire_period)
        ack_deadline = time.monotonic() + max(0.5, pulse_duration + 0.5)
        ack_seen = False
        while time.monotonic() < ack_deadline:
            try:
                msg = master.recv_match(blocking=False)
            except Exception:
                msg = None
            if msg is None:
                time.sleep(0.02)
                continue
            if msg.get_type() == "COMMAND_ACK":
                try:
                    cmd = int(msg.command)
                    if cmd in (MAV_CMD_DO_REPEAT_RELAY, MAV_CMD_DO_SET_RELAY):
                        print(f"[ACK] cmd={cmd} result={msg.result} "
                              f"(0=ACCEPTED, 2=DENIED, 4=FAILED, 5=UNSUPPORTED)")
                        if cmd == MAV_CMD_DO_REPEAT_RELAY:
                            ack_seen = True
                except Exception:
                    pass
            elif msg.get_type() == "STATUSTEXT":
                text = (getattr(msg, "text", "") or "").strip()
                if text:
                    print(f"[STATUSTEXT sev={getattr(msg, 'severity', '?')}] {text}")
        if not ack_seen:
            print("[FIRE] WARNING: no ACK for cmd=182 within "
                  f"{max(0.5, pulse_duration + 0.5):.2f}s "
                  f"(packet may have been lost or autopilot didn't respond)")
        else:
            print("[FIRE] done")
        try:
            master.close()
        except Exception:
            pass
        return 0

    master = None
    if not args.no_mavlink:
        try:
            from pymavlink import mavutil  # noqa: F401
        except ImportError:
            print("[MAVLINK] pymavlink not installed. Re-run with --no-mavlink for dry-run.")
            return 2
        from pymavlink import mavutil

        print(f"[MAVLINK] Connecting: {args.mavlink}")
        try:
            master = mavutil.mavlink_connection(args.mavlink)
            master.wait_heartbeat(timeout=args.heartbeat_timeout)
            print(f"[MAVLINK] heartbeat sysid={master.target_system} "
                  f"compid={master.target_component}")
            request_streams(master, args.stream_rate)
        except Exception as e:
            print(f"[MAVLINK] WARNING: connection/heartbeat failed: {e}")
            print("[MAVLINK] Continuing without transmission (panel will show no telemetry).")
            master = None
    else:
        print("[MAVLINK] Disabled via --no-mavlink (no commands will be sent)")

    if not sys.stdin.isatty():
        print("[INPUT] stdin is not a TTY; continuous keypress control disabled.")
        print("        Run interactively in a terminal for w/a/s/d control.")

    tel = Telemetry()
    pitch = clamp(args.initial_pitch, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
    yaw = clamp(args.initial_yaw, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
    step = float(args.step)

    send_interval = 1.0 / float(args.send_rate)
    redraw_interval = 1.0 / float(args.redraw_rate)
    heartbeat_send_interval = 1.0 / float(args.heartbeat_send_rate)

    firing = False
    fire_start_time: Optional[float] = None
    # MAV_CMD_DO_REPEAT_RELAY is fire-and-forget: ArduPilot owns the pulse timing,
    # so the script just tracks how long the burst should last for the UI banner.
    fire_burst_duration = float(args.fire_cycles) * float(args.fire_period)

    def fire_pulse() -> None:
        """Send one DO_REPEAT_RELAY burst per 'f' press. Exact shape of
        QGroundControl's 'Shoot Gun' action. NO double-send — two commands
        within the cycle window cause the autopilot to start overlapping pulse
        state machines, producing on/off/on/off chatter."""
        nonlocal firing, fire_start_time
        mode = "LIVE" if args.live_fire else "BLANK"
        if args.live_fire and master is not None:
            try:
                send_repeat_relay(master, args.fire_relay, args.fire_cycles, args.fire_period)
            except Exception as e:
                print(f"\n[FIRE ERROR] DO_REPEAT_RELAY failed: {e}", flush=True)
                return
        firing = True
        fire_start_time = time.monotonic()
        print(f"\n[{mode} FIRE PULSE] cmd=DO_REPEAT_RELAY(182) param1={args.fire_relay} "
              f"param2={args.fire_cycles} param3={args.fire_period:.2f}s "
              f"(autopilot will hold ON ~{fire_burst_duration/2:.2f}s)", flush=True)

    def fire_force_off() -> None:
        """Belt-and-suspenders safety: send DO_SET_RELAY(relay, 0) to drive the
        relay low immediately. The autopilot's repeat-relay timer should also
        reset on a fresh DO_SET_RELAY command."""
        nonlocal firing, fire_start_time
        if args.live_fire and master is not None:
            for _ in range(3):
                try:
                    send_relay(master, args.fire_relay, 0)
                except Exception:
                    break
                time.sleep(0.02)
        mode = "LIVE" if args.live_fire else "BLANK"
        print(f"\n[{mode} FIRE EMERGENCY OFF] cmd=DO_SET_RELAY(181) param1={args.fire_relay} param2=0",
              flush=True)
        firing = False
        fire_start_time = None

    def transmit() -> None:
        if master is None:
            tel.last_send_time = time.monotonic()
            tel.send_count += 1
            return
        try:
            send_mount_control(master, pitch, yaw)
            tel.last_send_err = None
        except Exception as e:
            tel.last_send_err = str(e)
        tel.last_send_time = time.monotonic()
        tel.send_count += 1

    sys.stdout.write(ANSI_CLEAR_SCREEN + ANSI_HOME + ANSI_HIDE_CURSOR)
    sys.stdout.flush()

    transmit()
    last_send = time.monotonic()
    last_heartbeat_send = last_send
    last_redraw = 0.0

    try:
        with RawTerminal() as term:
            running = True
            while running:
                ch = term.read_key()
                dirty = False
                while ch is not None:
                    if ch in ("q", "Q", "\x03"):
                        running = False
                        break
                    elif ch in ("w", "W"):
                        pitch = clamp(pitch + step, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
                        dirty = True
                    elif ch in ("s", "S"):
                        pitch = clamp(pitch - step, GIMBAL_PITCH_MIN_DEG, GIMBAL_PITCH_MAX_DEG)
                        dirty = True
                    elif ch in ("a", "A"):
                        yaw = clamp(yaw - step, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
                        dirty = True
                    elif ch in ("d", "D"):
                        yaw = clamp(yaw + step, GIMBAL_YAW_MIN_DEG, GIMBAL_YAW_MAX_DEG)
                        dirty = True
                    elif ch in ("c", "C"):
                        pitch = 0.0
                        yaw = 0.0
                        dirty = True
                    elif ch in ("+", "="):
                        step = min(STEP_MAX, step + 0.5)
                    elif ch in ("-", "_"):
                        step = max(STEP_MIN, step - 0.5)
                    elif ch in ("r", "R"):
                        dirty = True
                    elif ch == "f":
                        fire_pulse()
                    elif ch == "F":
                        fire_force_off()
                    ch = term.read_key()

                if master is not None:
                    for _ in range(500):
                        msg = master.recv_match(blocking=False)
                        if msg is None:
                            break
                        if msg.get_type() == "BAD_DATA":
                            continue
                        tel.ingest(msg)

                now = time.monotonic()
                if dirty and (now - last_send) >= send_interval:
                    transmit()
                    last_send = now
                    last_heartbeat_send = now
                elif (now - last_heartbeat_send) >= heartbeat_send_interval:
                    transmit()
                    last_heartbeat_send = now
                    last_send = now

                # MAV_CMD_DO_REPEAT_RELAY is fire-and-forget. Auto-clear the firing
                # flag once the autopilot's burst is done.
                if firing and fire_start_time is not None:
                    if (now - fire_start_time) >= fire_burst_duration:
                        firing = False
                        fire_start_time = None

                if (now - last_redraw) >= redraw_interval:
                    render(tel, pitch, yaw, step,
                           dry_run=args.no_mavlink,
                           master_ok=master is not None,
                           mav_string=args.mavlink if not args.no_mavlink else "(disabled)",
                           firing=firing,
                           fire_relay=args.fire_relay,
                           live_fire=args.live_fire,
                           fire_elapsed=(now - fire_start_time) if (firing and fire_start_time) else 0.0)
                    last_redraw = now

                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        # SAFETY: force the relay OFF on every exit path so the gun cannot
        # be left firing if the script crashes or is Ctrl-C'd while engaged.
        if args.live_fire and master is not None:
            for _ in range(3):
                try:
                    send_relay(master, args.fire_relay, 0)
                except Exception:
                    break
                time.sleep(0.02)
        sys.stdout.write(ANSI_SHOW_CURSOR + "\n")
        sys.stdout.flush()
        if args.live_fire:
            print(f"[FIRE] safety: relay {args.fire_relay} forced OFF on exit")
        if master is not None:
            try:
                master.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
