"""Shared aim-offset helper for the production auto scripts.

Why this exists
---------------
The water gun is mounted slightly to the right of the camera and pitched
differently to keep the camera's view clear. Propwash also nudges the water
stream during flight. The camera's "centered on target" pose therefore
points the gun a few cm right/up of the target, so it misses.

Concept
-------
The controller usually drives err_x/err_y toward 0 (target centered in
frame). With an aim offset, it drives err toward `(aim_x_norm, aim_y_norm)`
instead — i.e. the target ends up at that normalized position in the frame,
which puts the gun on the target.

Sign convention:
  positive aim_offset_x_norm  -> target appears RIGHT of center
                                 -> camera aims LEFT of target
                                 -> gun (right of camera) ends up ON target
  positive aim_offset_y_norm  -> target appears BELOW center
                                 -> camera aims ABOVE target
                                 -> water (after propwash drop) lands ON target

Files / CLI surface
-------------------
Default JSON: <repo_root>/gimbal_offset.json

CLI flags added by `add_aim_offset_args(parser)`:
  --aim-offset-x-norm  (float, default None  -> use file/zero)
  --aim-offset-y-norm  (float, default None  -> use file/zero)
  --aim-offset-file    (str,   default None  -> repo_root/gimbal_offset.json)
  --ignore-aim-offset  (flag,  default False -> force (0, 0))

Use `resolve_aim_offset_from_args(args)` after `parse_args()` to get the
effective (x_norm, y_norm) tuple. Then in the gimbal tracking math:

    aim_x, aim_y = resolve_aim_offset_from_args(args)
    adj_err_x = err_x - aim_x
    adj_err_y = err_y - aim_y
    # use adj_err_* for direction_label and the tracking law
    # use raw err_x/err_y during PHASE_VERIFY so the photo is on-target

JSON format (v1):
    {
      "version": 1,
      "aim_offset_x_norm": -0.03,
      "aim_offset_y_norm":  0.04,
      "notes": "drone hover 3m, water in tank, no propwash compensation",
      "calibrated_at": "2026-05-18T18:00:00Z"
    }
"""

import json
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_OFFSET_FILENAME = "gimbal_offset.json"
SCHEMA_VERSION = 1


def _repo_root() -> Path:
    # test/aim_offset.py -> test/ -> repo root
    return Path(__file__).resolve().parent.parent


def default_offset_path() -> Path:
    return _repo_root() / DEFAULT_OFFSET_FILENAME


def add_aim_offset_args(parser, group_title: str = "Gimbal aim offset") -> None:
    """Add the four standard aim-offset flags to an argparse parser.
    Safe to call multiple times only if the parser has not yet parsed.
    """
    g = parser.add_argument_group(group_title)
    g.add_argument(
        "--aim-offset-x-norm", type=float, default=None,
        help="Normalized x-offset where the controller should place the "
             "target in frame. + = right of center. None -> load from file "
             "or zero. Typical range [-0.10, +0.10].",
    )
    g.add_argument(
        "--aim-offset-y-norm", type=float, default=None,
        help="Normalized y-offset where the controller should place the "
             "target in frame. + = below center. None -> load from file "
             "or zero. Typical range [-0.10, +0.10].",
    )
    g.add_argument(
        "--aim-offset-file", type=str, default=None,
        help=f"Path to JSON file holding aim_offset_x_norm + "
             f"aim_offset_y_norm. Default <repo_root>/{DEFAULT_OFFSET_FILENAME}.",
    )
    g.add_argument(
        "--ignore-aim-offset", action="store_true",
        help="Force aim offset to (0, 0) regardless of file or CLI overrides. "
             "Used for A/B comparisons and to reproduce pre-calibration "
             "behavior of the auto scripts.",
    )


def load_aim_offset_file(path: Path) -> Optional[Tuple[float, float]]:
    """Load (x_norm, y_norm) from a JSON file. Returns None if the file does
    not exist; raises on malformed JSON or unknown schema version."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed aim-offset JSON at {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Aim-offset JSON at {path} must be an object")
    ver = data.get("version", 1)
    if ver != SCHEMA_VERSION:
        raise ValueError(
            f"Aim-offset JSON at {path} has version={ver}; only "
            f"version={SCHEMA_VERSION} is supported")
    try:
        x = float(data["aim_offset_x_norm"])
        y = float(data["aim_offset_y_norm"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"Aim-offset JSON at {path} missing aim_offset_x_norm / "
            f"aim_offset_y_norm: {e}") from e
    return x, y


def save_aim_offset_file(
    path: Path,
    x_norm: float,
    y_norm: float,
    notes: str = "",
    calibrated_at: Optional[str] = None,
) -> None:
    """Persist (x_norm, y_norm) to a JSON file with the standard schema."""
    payload = {
        "version": SCHEMA_VERSION,
        "aim_offset_x_norm": float(x_norm),
        "aim_offset_y_norm": float(y_norm),
        "notes": notes,
        "calibrated_at": calibrated_at,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_aim_offset_from_args(args) -> Tuple[float, float]:
    """Return the effective (x_norm, y_norm) given CLI args. Priority:
       1. --ignore-aim-offset      -> (0, 0)
       2. --aim-offset-{x,y}-norm  -> CLI values (zeros for unset axis)
       3. --aim-offset-file (or default path)  -> file contents
       4. default                   -> (0, 0)
    Prints a single line summarizing the source for log auditability.
    """
    if getattr(args, "ignore_aim_offset", False):
        print("[AIM-OFFSET] ignored (--ignore-aim-offset); using (0, 0)",
              flush=True)
        return 0.0, 0.0

    cli_x = getattr(args, "aim_offset_x_norm", None)
    cli_y = getattr(args, "aim_offset_y_norm", None)
    if cli_x is not None or cli_y is not None:
        x = float(cli_x) if cli_x is not None else 0.0
        y = float(cli_y) if cli_y is not None else 0.0
        print(f"[AIM-OFFSET] CLI override: ({x:+.4f}, {y:+.4f})", flush=True)
        return x, y

    file_arg = getattr(args, "aim_offset_file", None)
    path = Path(file_arg) if file_arg else default_offset_path()
    loaded = load_aim_offset_file(path)
    if loaded is not None:
        x, y = loaded
        print(f"[AIM-OFFSET] loaded from {path}: ({x:+.4f}, {y:+.4f})",
              flush=True)
        return x, y

    print(f"[AIM-OFFSET] no file at {path}; using (0, 0)", flush=True)
    return 0.0, 0.0
