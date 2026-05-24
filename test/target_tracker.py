"""Shared, persistent Task-2 target counter.

Both the autonomous engagement scripts (final_auto / gimbal_back_up) and the
always-on manual photo service read/advance the SAME counter so target numbers
stay in true extinguishing order across autonomous and manual captures
(Task 2 §f: "target numbers go in order of extinguishing").

Backed by a single JSON file (default ``/images/target_state.json``) guarded by
an ``fcntl`` exclusive lock for safe concurrent access by multiple processes.

Semantics:
  ``next_target`` = the number to stamp on the target currently being engaged.
  ``current_target()`` returns it (use for the photo filename).
  ``advance_target()`` is called ONLY when a target is confirmed extinguished
  (wetness check == WETTED); it returns the number that was just used and bumps
  ``next_target`` so the next engagement uses the next integer. A miss/retry
  therefore reuses the same number (correct: same physical target).
"""

import json
import os
import fcntl
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_STATE = {"team_name": "unknown", "next_target": 1, "history": [], "boot_id": ""}


def _read_boot_id():
    """Kernel boot id — changes on every power cycle / reboot, stays constant
    across process restarts within the same boot."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:
        return None


class TargetTracker:
    def __init__(self, state_path="/images/target_state.json",
                 team_name=None, start_target=None, reset_per_boot=True):
        self.path = Path(state_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_next = 1
        self._cached_team = "unknown"
        boot_id = _read_boot_id()
        # Seed / reconcile on construction. The FIRST process to build a tracker
        # after a power cycle resets the counter to 1 (boot id changed); restarts
        # within the same boot keep the running count.
        with self._locked() as st:
            if reset_per_boot and boot_id and st.get("boot_id") != boot_id:
                st["next_target"] = 1
                st["history"] = []
            if boot_id:
                st["boot_id"] = boot_id
            if team_name is not None:
                st["team_name"] = team_name
            if start_target is not None:  # explicit override wins over boot reset
                st["next_target"] = int(start_target)
            self._cached_next = int(st["next_target"])
            self._cached_team = st["team_name"]

    @contextmanager
    def _locked(self):
        """Open-create the state file, take an exclusive lock, yield the parsed
        dict, then write it back. Read-only callers harmlessly rewrite the same
        content."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.lseek(fd, 0, 0)
            raw = os.read(fd, 4_000_000).decode("utf-8", "replace")
            try:
                st = json.loads(raw) if raw.strip() else dict(DEFAULT_STATE)
            except Exception:
                st = dict(DEFAULT_STATE)
            for k, v in DEFAULT_STATE.items():
                st.setdefault(k, v if not isinstance(v, list) else [])
            yield st
            out = json.dumps(st, indent=2).encode("utf-8")
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
            os.write(fd, out)
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def get_team(self):
        with self._locked() as st:
            self._cached_team = st["team_name"]
            return st["team_name"]

    def set_team(self, name):
        with self._locked() as st:
            st["team_name"] = name
            self._cached_team = name

    def current_target(self):
        """The number to stamp on the target currently being engaged (live read
        with file lock — use for filenames)."""
        with self._locked() as st:
            self._cached_next = int(st["next_target"])
            return self._cached_next

    def cached_current(self):
        """Last-known target number WITHOUT file I/O — for hot OSD/log paths.
        Reflects this process's view (manual service runs in a separate process;
        its advances aren't seen here, which is fine for cosmetic display)."""
        return self._cached_next

    def cached_team(self):
        return self._cached_team

    def advance_target(self, reason=""):
        """Confirm the current target extinguished. Returns the number just
        used; bumps the counter for the next engagement."""
        with self._locked() as st:
            cur = int(st["next_target"])
            st["next_target"] = cur + 1
            st.setdefault("history", []).append({
                "target": cur,
                "reason": reason,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            self._cached_next = cur + 1
            return cur
