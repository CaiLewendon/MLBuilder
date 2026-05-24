#!/usr/bin/env python3
"""Quick MAVLink tap — shows what the Pi is actually receiving on its endpoint.

Run it, then press your QGC photo buttons. It reports:
  - who is on the bus (sysid,compid of every sender) -> is the flight computer there?
  - every COMMAND_LONG / COMMAND_INT / COMMAND_ACK with cmd+param -> is the GS
    sending, and does it reach the Pi?
  - a tally of all message types -> is telemetry flowing at all?

Usage:  python3 ~/mavlink_tap.py            (defaults to tcp:10.42.0.1:5760)
        python3 ~/mavlink_tap.py udpin:0.0.0.0:14550
Stop the photo service first so they don't both read the endpoint:
        sudo systemctl stop mavlink_photo_service
"""
import sys
import threading
import time

from pymavlink import mavutil

endpoint = sys.argv[1] if len(sys.argv) > 1 else "tcp:10.42.0.1:5760"
print(f"[TAP] connecting {endpoint} ...", flush=True)
# Announce ourselves as a component of the vehicle (sys 1) so mavlink-router
# delivers vehicle-targeted COMMAND_LONGs (the QGC buttons) to this endpoint.
m = mavutil.mavlink_connection(endpoint, source_system=1, source_component=191)


def _heartbeat_loop():
    while True:
        try:
            m.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        except Exception:
            pass
        time.sleep(1.0)


threading.Thread(target=_heartbeat_loop, daemon=True).start()
print("[TAP] announcing heartbeat as sys=1 comp=191 (so the router routes "
      "vehicle-targeted commands here)", flush=True)
hb = m.wait_heartbeat(timeout=15)
if hb is None:
    print("[TAP] NO heartbeat in 15s — nothing is talking on this endpoint.", flush=True)
else:
    print(f"[TAP] first heartbeat: srcSys={hb.get_srcSystem()} "
          f"srcComp={hb.get_srcComponent()} type={getattr(hb,'type','?')}", flush=True)
print("[TAP] listening 60s — PRESS YOUR QGC PHOTO BUTTONS NOW (Ctrl-C to stop).",
      flush=True)

seen = {}
senders = set()
t0 = time.time()
try:
    while time.time() - t0 < 60:
        msg = m.recv_match(blocking=True, timeout=1)
        if msg is None:
            continue
        typ = msg.get_type()
        if typ == "BAD_DATA":
            continue
        seen[typ] = seen.get(typ, 0) + 1
        senders.add((msg.get_srcSystem(), msg.get_srcComponent()))
        if typ in ("COMMAND_LONG", "COMMAND_INT", "COMMAND_ACK"):
            print(f"  >>> {typ}: cmd={getattr(msg,'command','?')} "
                  f"p1={getattr(msg,'param1','?')} p2={getattr(msg,'param2','?')} "
                  f"from sys={msg.get_srcSystem()} comp={msg.get_srcComponent()}",
                  flush=True)
except KeyboardInterrupt:
    pass

print("\n[TAP] senders (sysid,compid) seen:", sorted(senders), flush=True)
print("[TAP] message types + counts:", flush=True)
for k in sorted(seen):
    print(f"    {k}: {seen[k]}", flush=True)
