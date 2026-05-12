#!/usr/bin/env python3
"""
Full-suite MAVLink parity validator for:
  - tf_live_infrence_drone_simulation.py
  - tf_live_infrence_gimbal_simulation.py

Default mode is telemetry-only (passive). Optional active gimbal probe is guarded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None

# Single editable connection string as requested.
CONNECTION_STRING = "tcp:10.42.0.1:5760"

DEFAULT_DURATION_SEC = 60.0
DEFAULT_MODE = "passive"
DEFAULT_STRICTNESS = "balanced"

MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_MOUNT_MODE_MAVLINK_TARGETING = 2

# Bounded active gimbal probe values (small commands for safe bench checks).
GIMBAL_PROBE_SEQUENCE_DEG = [
    {"pitch": 0.0, "yaw": 0.0},
    {"pitch": 2.0, "yaw": 2.0},
    {"pitch": -2.0, "yaw": -2.0},
    {"pitch": 0.0, "yaw": 0.0},
]
GIMBAL_ACK_TIMEOUT_SEC = 2.0
GIMBAL_STEP_DELAY_SEC = 0.25

ORIENTATION_NAMES: dict[int, str] = {
    0: "forward",
    2: "right",
    4: "back",
    6: "left",
    24: "up",
    25: "down",
}
PRIMARY_FORWARD_ORIENTATION = 0


STRICTNESS_PROFILES: dict[str, dict[str, float]] = {
    "lenient": {
        "heartbeat_min_hz": 0.3,
        "attitude_min_hz": 2.0,
        "local_position_min_hz": 1.5,
        "distance_forward_min_hz": 1.0,
        "dropout_warn_sec": 3.0,
        "dropout_fail_sec": 6.0,
    },
    "balanced": {
        "heartbeat_min_hz": 0.5,
        "attitude_min_hz": 5.0,
        "local_position_min_hz": 3.0,
        "distance_forward_min_hz": 2.0,
        "dropout_warn_sec": 2.0,
        "dropout_fail_sec": 4.0,
    },
    "strict": {
        "heartbeat_min_hz": 0.8,
        "attitude_min_hz": 8.0,
        "local_position_min_hz": 5.0,
        "distance_forward_min_hz": 4.0,
        "dropout_warn_sec": 1.0,
        "dropout_fail_sec": 2.0,
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sanitize_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def verdict_rank(verdict: str) -> int:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return order.get(verdict, 2)


def combine_verdict(*verdicts: str) -> str:
    worst = "PASS"
    for v in verdicts:
        if verdict_rank(v) > verdict_rank(worst):
            worst = v
    return worst


@dataclass
class StreamStats:
    name: str
    count: int = 0
    first_ts: float | None = None
    last_ts: float | None = None
    max_gap_sec: float = 0.0
    last_value: dict[str, float] = field(default_factory=dict)
    min_value: dict[str, float] = field(default_factory=dict)
    max_value: dict[str, float] = field(default_factory=dict)
    invalid_value_count: int = 0

    def update(self, ts: float, fields: dict[str, float | None]) -> None:
        self.count += 1
        if self.first_ts is None:
            self.first_ts = ts
        if self.last_ts is not None:
            self.max_gap_sec = max(self.max_gap_sec, ts - self.last_ts)
        self.last_ts = ts

        for key, raw_value in fields.items():
            value = sanitize_float(raw_value)
            if value is None:
                self.invalid_value_count += 1
                continue
            self.last_value[key] = value
            if key not in self.min_value:
                self.min_value[key] = value
                self.max_value[key] = value
            else:
                self.min_value[key] = min(self.min_value[key], value)
                self.max_value[key] = max(self.max_value[key], value)

    def rate_hz(self, duration_sec: float) -> float:
        if duration_sec <= 0.0:
            return 0.0
        return float(self.count) / float(duration_sec)

    def staleness_sec(self, now_ts: float) -> float:
        if self.last_ts is None:
            return float("inf")
        return max(0.0, now_ts - self.last_ts)

    def to_dict(self, duration_sec: float, now_ts: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "rate_hz": self.rate_hz(duration_sec),
            "staleness_sec": self.staleness_sec(now_ts),
            "max_gap_sec": self.max_gap_sec,
            "invalid_value_count": self.invalid_value_count,
            "last_value": self.last_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


@dataclass
class EvalResult:
    verdict: str
    blockers: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "blockers": self.blockers,
            "warnings": self.warnings,
        }


def evaluate_required_stream(
    stream: StreamStats,
    min_hz: float,
    dropout_warn_sec: float,
    dropout_fail_sec: float,
    duration_sec: float,
    now_ts: float,
    label: str,
) -> EvalResult:
    blockers: list[str] = []
    warnings: list[str] = []

    rate = stream.rate_hz(duration_sec)
    stale = stream.staleness_sec(now_ts)

    if stream.count <= 0:
        blockers.append(f"{label}: stream missing")
    if rate < min_hz:
        blockers.append(f"{label}: low rate {rate:.2f}Hz < {min_hz:.2f}Hz")
    if stale >= dropout_fail_sec:
        blockers.append(f"{label}: stale for {stale:.2f}s (>= fail {dropout_fail_sec:.2f}s)")
    elif stale >= dropout_warn_sec:
        warnings.append(f"{label}: stale for {stale:.2f}s (>= warn {dropout_warn_sec:.2f}s)")

    if stream.invalid_value_count > 0:
        warnings.append(f"{label}: {stream.invalid_value_count} invalid value fields observed")

    verdict = "PASS"
    if blockers:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARN"

    return EvalResult(verdict=verdict, blockers=blockers, warnings=warnings)


def evaluate_distance_sensor_sanity(stream: StreamStats, label: str) -> EvalResult:
    blockers: list[str] = []
    warnings: list[str] = []

    last = stream.last_value
    cur = last.get("current_distance")
    mn = last.get("min_distance")
    mx = last.get("max_distance")

    if cur is not None:
        if cur < 0:
            blockers.append(f"{label}: current_distance < 0")
        if mn is not None and cur < mn:
            warnings.append(f"{label}: current_distance {cur:.2f} < min_distance {mn:.2f}")
        if mx is not None and cur > mx:
            warnings.append(f"{label}: current_distance {cur:.2f} > max_distance {mx:.2f}")

    verdict = "PASS"
    if blockers:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARN"
    return EvalResult(verdict=verdict, blockers=blockers, warnings=warnings)


def send_mount_control(master: mavutil.mavfile, pitch_deg: float, yaw_deg: float) -> None:
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


def wait_for_command_ack(
    master: mavutil.mavfile,
    command_id: int,
    timeout_sec: float,
) -> tuple[bool, dict[str, Any]]:
    t_end = time.monotonic() + timeout_sec
    while time.monotonic() < t_end:
        msg = master.recv_match(blocking=True, timeout=0.2)
        if msg is None:
            continue
        if msg.get_type() != "COMMAND_ACK":
            continue
        cmd = int(getattr(msg, "command", -1))
        if cmd != command_id:
            continue
        result = int(getattr(msg, "result", -1))
        return True, {
            "command": cmd,
            "result": result,
            "result_name": mavutil.mavlink.enums["MAV_RESULT"].get(result, None).name
            if result in mavutil.mavlink.enums["MAV_RESULT"]
            else f"UNKNOWN_{result}",
        }
    return False, {"command": command_id, "timeout_sec": timeout_sec}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pi_mavlink_fullsuite_test",
        description="Full-suite MAVLink compatibility validator for tf_live_* simulation logic",
    )
    parser.add_argument(
        "--mode",
        choices=["passive", "active-gimbal", "active-all"],
        default=DEFAULT_MODE,
        help="Validation mode (default: passive)",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help="Telemetry sampling duration (seconds)",
    )
    parser.add_argument(
        "--strictness",
        choices=["lenient", "balanced", "strict"],
        default=DEFAULT_STRICTNESS,
        help="Threshold profile for pass/fail",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional JSON output path (default: timestamped file in current directory)",
    )
    parser.add_argument(
        "--enable-active-gimbal",
        action="store_true",
        help="Required guard flag to run active gimbal probe modes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_sec <= 0.0:
        raise SystemExit("--duration-sec must be > 0")
    if mavutil is None:
        raise SystemExit(
            "pymavlink is not installed. Install with `pip install pymavlink` "
            "or use the project requirements on the Pi target."
        )

    profile = STRICTNESS_PROFILES[args.strictness]

    if args.mode in {"active-gimbal", "active-all"} and not args.enable_active_gimbal:
        raise SystemExit(
            "Refusing active mode without --enable-active-gimbal. "
            "Use passive mode or explicitly acknowledge active probe."
        )

    print("[INFO] Full-suite MAVLink parity validator")
    print(
        "[INFO] Safety: no drone flight movement commands are sent in any mode; "
        "only optional bounded gimbal mount probes may be sent."
    )
    print(f"[INFO] Connection string: {CONNECTION_STRING}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Strictness: {args.strictness}")
    print(f"[INFO] Telemetry sample duration: {args.duration_sec:.1f}s")

    start_wall = utc_now_iso()
    start_mono = time.monotonic()

    master = mavutil.mavlink_connection(CONNECTION_STRING)
    master.wait_heartbeat(timeout=15)
    heartbeat_recv_ts = time.monotonic()
    print(
        f"[INFO] Heartbeat received: sysid={master.target_system} compid={master.target_component}"
    )

    streams: dict[str, StreamStats] = {
        "HEARTBEAT": StreamStats("HEARTBEAT"),
        "ATTITUDE": StreamStats("ATTITUDE"),
        "LOCAL_POSITION_NED": StreamStats("LOCAL_POSITION_NED"),
        "GLOBAL_POSITION_INT": StreamStats("GLOBAL_POSITION_INT"),
        "VFR_HUD": StreamStats("VFR_HUD"),
        "RANGEFINDER": StreamStats("RANGEFINDER"),
        "SYS_STATUS": StreamStats("SYS_STATUS"),
        "AUTOPILOT_VERSION": StreamStats("AUTOPILOT_VERSION"),
    }

    distance_by_orientation: dict[int, StreamStats] = {}
    unknown_distance_orientations: set[int] = set()

    sample_end = time.monotonic() + args.duration_sec
    while time.monotonic() < sample_end:
        msg = master.recv_match(blocking=True, timeout=0.2)
        if msg is None:
            continue
        ts = time.monotonic()
        mtype = msg.get_type()

        if mtype == "BAD_DATA":
            continue
        if mtype == "HEARTBEAT":
            streams["HEARTBEAT"].update(
                ts,
                {
                    "type": getattr(msg, "type", None),
                    "autopilot": getattr(msg, "autopilot", None),
                    "base_mode": getattr(msg, "base_mode", None),
                    "custom_mode": getattr(msg, "custom_mode", None),
                    "system_status": getattr(msg, "system_status", None),
                },
            )
        elif mtype == "ATTITUDE":
            streams["ATTITUDE"].update(
                ts,
                {
                    "roll": getattr(msg, "roll", None),
                    "pitch": getattr(msg, "pitch", None),
                    "yaw": getattr(msg, "yaw", None),
                    "rollspeed": getattr(msg, "rollspeed", None),
                    "pitchspeed": getattr(msg, "pitchspeed", None),
                    "yawspeed": getattr(msg, "yawspeed", None),
                },
            )
        elif mtype == "LOCAL_POSITION_NED":
            streams["LOCAL_POSITION_NED"].update(
                ts,
                {
                    "x": getattr(msg, "x", None),
                    "y": getattr(msg, "y", None),
                    "z": getattr(msg, "z", None),
                    "vx": getattr(msg, "vx", None),
                    "vy": getattr(msg, "vy", None),
                    "vz": getattr(msg, "vz", None),
                },
            )
        elif mtype == "GLOBAL_POSITION_INT":
            streams["GLOBAL_POSITION_INT"].update(
                ts,
                {
                    "lat": getattr(msg, "lat", None),
                    "lon": getattr(msg, "lon", None),
                    "relative_alt": getattr(msg, "relative_alt", None),
                    "hdg": getattr(msg, "hdg", None),
                },
            )
        elif mtype == "VFR_HUD":
            streams["VFR_HUD"].update(
                ts,
                {
                    "heading": getattr(msg, "heading", None),
                    "climb": getattr(msg, "climb", None),
                    "groundspeed": getattr(msg, "groundspeed", None),
                },
            )
        elif mtype == "RANGEFINDER":
            streams["RANGEFINDER"].update(
                ts,
                {
                    "distance": getattr(msg, "distance", None),
                    "voltage": getattr(msg, "voltage", None),
                },
            )
        elif mtype == "SYS_STATUS":
            streams["SYS_STATUS"].update(
                ts,
                {
                    "load": getattr(msg, "load", None),
                    "voltage_battery": getattr(msg, "voltage_battery", None),
                    "battery_remaining": getattr(msg, "battery_remaining", None),
                },
            )
        elif mtype == "AUTOPILOT_VERSION":
            streams["AUTOPILOT_VERSION"].update(
                ts,
                {
                    "capabilities": getattr(msg, "capabilities", None),
                },
            )
        elif mtype == "DISTANCE_SENSOR":
            orientation = int(getattr(msg, "orientation", -1))
            if orientation not in distance_by_orientation:
                label = ORIENTATION_NAMES.get(orientation, f"orientation_{orientation}")
                distance_by_orientation[orientation] = StreamStats(f"DISTANCE_SENSOR[{label}]")
            if orientation not in ORIENTATION_NAMES:
                unknown_distance_orientations.add(orientation)

            distance_by_orientation[orientation].update(
                ts,
                {
                    "current_distance": getattr(msg, "current_distance", None),
                    "min_distance": getattr(msg, "min_distance", None),
                    "max_distance": getattr(msg, "max_distance", None),
                    "covariance": getattr(msg, "covariance", None),
                },
            )

    end_mono = time.monotonic()
    duration = end_mono - start_mono
    now_ts = end_mono

    # Required stream evaluations mapped to simulation logic dependencies.
    hb_eval = evaluate_required_stream(
        stream=streams["HEARTBEAT"],
        min_hz=profile["heartbeat_min_hz"],
        dropout_warn_sec=profile["dropout_warn_sec"],
        dropout_fail_sec=profile["dropout_fail_sec"],
        duration_sec=duration,
        now_ts=now_ts,
        label="HEARTBEAT",
    )
    att_eval = evaluate_required_stream(
        stream=streams["ATTITUDE"],
        min_hz=profile["attitude_min_hz"],
        dropout_warn_sec=profile["dropout_warn_sec"],
        dropout_fail_sec=profile["dropout_fail_sec"],
        duration_sec=duration,
        now_ts=now_ts,
        label="ATTITUDE",
    )
    lpn_eval = evaluate_required_stream(
        stream=streams["LOCAL_POSITION_NED"],
        min_hz=profile["local_position_min_hz"],
        dropout_warn_sec=profile["dropout_warn_sec"],
        dropout_fail_sec=profile["dropout_fail_sec"],
        duration_sec=duration,
        now_ts=now_ts,
        label="LOCAL_POSITION_NED",
    )

    forward_stream = distance_by_orientation.get(PRIMARY_FORWARD_ORIENTATION)
    if forward_stream is None:
        forward_eval = EvalResult(
            verdict="FAIL",
            blockers=["DISTANCE_SENSOR[forward]: stream missing"],
            warnings=[],
        )
        forward_sanity = EvalResult(verdict="FAIL", blockers=[], warnings=[])
    else:
        forward_eval = evaluate_required_stream(
            stream=forward_stream,
            min_hz=profile["distance_forward_min_hz"],
            dropout_warn_sec=profile["dropout_warn_sec"],
            dropout_fail_sec=profile["dropout_fail_sec"],
            duration_sec=duration,
            now_ts=now_ts,
            label="DISTANCE_SENSOR[forward]",
        )
        forward_sanity = evaluate_distance_sensor_sanity(forward_stream, "DISTANCE_SENSOR[forward]")

    # Drone-phase compatibility results.
    centering_verdict = combine_verdict(hb_eval.verdict, att_eval.verdict)
    approach_hold_verdict = combine_verdict(
        hb_eval.verdict, att_eval.verdict, forward_eval.verdict, forward_sanity.verdict
    )
    altitude_adjust_verdict = combine_verdict(hb_eval.verdict, lpn_eval.verdict)
    final_hold_verdict = combine_verdict(
        hb_eval.verdict,
        att_eval.verdict,
        forward_eval.verdict,
        lpn_eval.verdict,
        forward_sanity.verdict,
    )

    phase_results: dict[str, EvalResult] = {
        "CENTERING": EvalResult(
            verdict=centering_verdict,
            blockers=hb_eval.blockers + att_eval.blockers,
            warnings=hb_eval.warnings + att_eval.warnings,
        ),
        "APPROACH": EvalResult(
            verdict=approach_hold_verdict,
            blockers=hb_eval.blockers + att_eval.blockers + forward_eval.blockers + forward_sanity.blockers,
            warnings=hb_eval.warnings + att_eval.warnings + forward_eval.warnings + forward_sanity.warnings,
        ),
        "HOLD": EvalResult(
            verdict=approach_hold_verdict,
            blockers=hb_eval.blockers + att_eval.blockers + forward_eval.blockers + forward_sanity.blockers,
            warnings=hb_eval.warnings + att_eval.warnings + forward_eval.warnings + forward_sanity.warnings,
        ),
        "LOCKED_HOLD": EvalResult(
            verdict=approach_hold_verdict,
            blockers=hb_eval.blockers + att_eval.blockers + forward_eval.blockers + forward_sanity.blockers,
            warnings=hb_eval.warnings + att_eval.warnings + forward_eval.warnings + forward_sanity.warnings,
        ),
        "ALTITUDE_ADJUST": EvalResult(
            verdict=altitude_adjust_verdict,
            blockers=hb_eval.blockers + lpn_eval.blockers,
            warnings=hb_eval.warnings + lpn_eval.warnings,
        ),
        "FINAL_HOLD": EvalResult(
            verdict=final_hold_verdict,
            blockers=(
                hb_eval.blockers
                + att_eval.blockers
                + forward_eval.blockers
                + lpn_eval.blockers
                + forward_sanity.blockers
            ),
            warnings=(
                hb_eval.warnings
                + att_eval.warnings
                + forward_eval.warnings
                + lpn_eval.warnings
                + forward_sanity.warnings
            ),
        ),
    }

    drone_suite_verdict = "PASS"
    for phase in phase_results.values():
        drone_suite_verdict = combine_verdict(drone_suite_verdict, phase.verdict)

    # Gimbal passive compatibility.
    gimbal_passive_blockers: list[str] = []
    gimbal_passive_warnings: list[str] = []

    gimbal_passive_verdict = combine_verdict(hb_eval.verdict, att_eval.verdict)
    if master.target_system <= 0 or master.target_component <= 0:
        gimbal_passive_blockers.append(
            f"Invalid target identifiers sysid={master.target_system} compid={master.target_component}"
        )
        gimbal_passive_verdict = "FAIL"

    mount_related_present = any(
        name in streams and streams[name].count > 0
        for name in ("AUTOPILOT_VERSION", "SYS_STATUS")
    )
    if not mount_related_present:
        gimbal_passive_warnings.append(
            "No AUTOPILOT_VERSION/SYS_STATUS received; mount capability context may be incomplete"
        )
        gimbal_passive_verdict = combine_verdict(gimbal_passive_verdict, "WARN")

    gimbal_sign_report = {
        "simulation_mapping": {
            "image_x_positive": "right",
            "image_y_positive": "down",
            "gimbal_yaw_positive": "clockwise/right",
            "gimbal_pitch_positive": "up",
        },
        "telemetry_observed": {
            "attitude_yaw_last_rad": streams["ATTITUDE"].last_value.get("yaw"),
            "attitude_yawspeed_last_rad_s": streams["ATTITUDE"].last_value.get("yawspeed"),
        },
        "note": "Passive mode validates observability and mapping context; physical mount sign must be confirmed with active probe.",
    }

    gimbal_passive_eval = EvalResult(
        verdict=gimbal_passive_verdict,
        blockers=gimbal_passive_blockers + hb_eval.blockers + att_eval.blockers,
        warnings=gimbal_passive_warnings + hb_eval.warnings + att_eval.warnings,
    )

    # Optional active gimbal probe.
    active_probe_data: dict[str, Any] = {
        "mode_requested": args.mode,
        "enabled_guard": bool(args.enable_active_gimbal),
        "attempted": False,
        "acks": [],
        "verdict": "PASS",
        "blockers": [],
        "warnings": [],
    }
    if args.mode in {"active-gimbal", "active-all"}:
        active_probe_data["attempted"] = True
        for step in GIMBAL_PROBE_SEQUENCE_DEG:
            send_mount_control(master, pitch_deg=step["pitch"], yaw_deg=step["yaw"])
            ack_ok, ack = wait_for_command_ack(
                master, command_id=MAV_CMD_DO_MOUNT_CONTROL, timeout_sec=GIMBAL_ACK_TIMEOUT_SEC
            )
            active_probe_data["acks"].append(
                {
                    "pitch_deg": step["pitch"],
                    "yaw_deg": step["yaw"],
                    "ack_ok": ack_ok,
                    "ack": ack,
                }
            )
            if not ack_ok:
                active_probe_data["blockers"].append(
                    f"Missing COMMAND_ACK for MAV_CMD_DO_MOUNT_CONTROL (pitch={step['pitch']}, yaw={step['yaw']})"
                )
                break
            result_code = int(ack.get("result", -1))
            # Accepted (0) or temporarily rejected-in-flight is still reported; strict fail only for non-accept.
            if result_code != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                active_probe_data["warnings"].append(
                    f"ACK result {ack.get('result_name')} for pitch={step['pitch']} yaw={step['yaw']}"
                )
            time.sleep(GIMBAL_STEP_DELAY_SEC)

        # Explicit neutral/reset command at end.
        send_mount_control(master, pitch_deg=0.0, yaw_deg=0.0)
        neutral_ok, neutral_ack = wait_for_command_ack(
            master, command_id=MAV_CMD_DO_MOUNT_CONTROL, timeout_sec=GIMBAL_ACK_TIMEOUT_SEC
        )
        active_probe_data["neutral_ack"] = {"ok": neutral_ok, "ack": neutral_ack}
        if not neutral_ok:
            active_probe_data["blockers"].append("No ACK on neutral/reset mount command")

        if active_probe_data["blockers"]:
            active_probe_data["verdict"] = "FAIL"
        elif active_probe_data["warnings"]:
            active_probe_data["verdict"] = "WARN"
        else:
            active_probe_data["verdict"] = "PASS"

    gimbal_active_verdict = active_probe_data["verdict"]
    if args.mode == "passive":
        gimbal_active_verdict = "PASS"

    overall_verdict = combine_verdict(drone_suite_verdict, gimbal_passive_eval.verdict, gimbal_active_verdict)

    stream_dump = {name: st.to_dict(duration, now_ts) for name, st in streams.items()}
    distance_dump: dict[str, Any] = {}
    for orientation, st in distance_by_orientation.items():
        key = ORIENTATION_NAMES.get(orientation, f"orientation_{orientation}")
        distance_dump[key] = st.to_dict(duration, now_ts)

    # Directional range inventory for diagnostics.
    directional_inventory = {
        name: {"present": False, "count": 0, "rate_hz": 0.0}
        for name in ("forward", "back", "left", "right", "up", "down")
    }
    for orientation, label in ORIENTATION_NAMES.items():
        if label not in directional_inventory:
            continue
        st = distance_by_orientation.get(orientation)
        if st is None:
            continue
        directional_inventory[label]["present"] = st.count > 0
        directional_inventory[label]["count"] = st.count
        directional_inventory[label]["rate_hz"] = st.rate_hz(duration)

    report = {
        "meta": {
            "generated_at_utc": utc_now_iso(),
            "start_wall_utc": start_wall,
            "duration_sec": duration,
            "mode": args.mode,
            "strictness": args.strictness,
            "connection_string": CONNECTION_STRING,
            "target_system": master.target_system,
            "target_component": master.target_component,
            "heartbeat_received_sec_after_start": heartbeat_recv_ts - start_mono,
        },
        "verdicts": {
            "overall": overall_verdict,
            "drone_logic_compatibility": drone_suite_verdict,
            "gimbal_logic_compatibility_passive": gimbal_passive_eval.verdict,
            "gimbal_logic_compatibility_active_probe": gimbal_active_verdict,
        },
        "phase_compatibility": {k: v.to_dict() for k, v in phase_results.items()},
        "gimbal_passive": {
            **gimbal_passive_eval.to_dict(),
            "sign_mapping_report": gimbal_sign_report,
        },
        "gimbal_active_probe": active_probe_data,
        "streams": stream_dump,
        "distance_sensor": {
            "by_orientation": distance_dump,
            "inventory": directional_inventory,
            "unknown_orientations": sorted(list(unknown_distance_orientations)),
            "primary_forward_orientation": PRIMARY_FORWARD_ORIENTATION,
        },
    }

    # Console summary
    print("\n=== Full-Suite Verdicts ===")
    print(f"Overall: {overall_verdict}")
    print(f"Drone Logic Compatibility: {drone_suite_verdict}")
    print(f"Gimbal Logic Compatibility (Passive): {gimbal_passive_eval.verdict}")
    print(f"Gimbal Logic Compatibility (Active Probe): {gimbal_active_verdict}")

    print("\n=== Phase Compatibility ===")
    for phase_name in (
        "CENTERING",
        "APPROACH",
        "HOLD",
        "LOCKED_HOLD",
        "ALTITUDE_ADJUST",
        "FINAL_HOLD",
    ):
        phase = phase_results[phase_name]
        print(f"{phase_name}: {phase.verdict}")
        for b in phase.blockers:
            print(f"  BLOCKER: {b}")
        for w in phase.warnings:
            print(f"  WARN: {w}")

    print("\n=== Directional Distance Sensor Inventory ===")
    for direction in ("forward", "back", "left", "right", "up", "down"):
        item = directional_inventory[direction]
        print(
            f"{direction:>7}: present={item['present']} count={item['count']} "
            f"rate_hz={item['rate_hz']:.2f}"
        )

    if active_probe_data["attempted"]:
        print("\n=== Active Gimbal Probe ===")
        print(f"Verdict: {active_probe_data['verdict']}")
        for a in active_probe_data["acks"]:
            print(
                "  step "
                f"pitch={a['pitch_deg']:+.1f} yaw={a['yaw_deg']:+.1f} "
                f"ack_ok={a['ack_ok']} ack={a['ack']}"
            )
        for b in active_probe_data["blockers"]:
            print(f"  BLOCKER: {b}")
        for w in active_probe_data["warnings"]:
            print(f"  WARN: {w}")

    json_out_path = args.json_out.strip()
    if not json_out_path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_out_path = f"mavlink_fullsuite_report_{stamp}.json"

    out_path = Path(json_out_path).expanduser().resolve()
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[INFO] JSON report written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
