"""Centralized Task-2 photo saving.

All autonomous (final_auto / gimbal_back_up) and manual (mavlink_photo_service)
captures route through here so filenames + folder layout are identical and the
``/images`` tree stays Google-Drive-sync friendly.

Layout:  <out_root>/Task_2_<team>_target_<#>_<ts>.jpg   (flat — all cameras,
autonomous + manual, in one folder for Google-Drive sync. out_root=/images.)

The filename mirrors the legacy ``do_verify_capture`` convention
(Task_2_<team>_target_<#>_<ts>.jpg, JPEG quality 95) so nothing downstream
changes shape — only the destination directory is centralized per camera.
"""

import time
from pathlib import Path

import cv2


def save_target_photo(frame, camera, team, target_num,
                      out_root="/images", jpeg_quality=95, ts=None):
    """Save ``frame`` as the canonical Task-2 photo for ``camera``.

    Returns the written ``Path``. Raises ``IOError`` only if cv2 fails to
    encode/write — callers should treat a save failure as loud but must never
    let it gate a real extinguish declaration (the operator can re-shoot).
    """
    # Flat layout, exact Task-2 submission title: Task_2_<team>_target_<#>.jpg
    # (no timestamp). Every photo — gun_high, front_high, autonomous, manual —
    # goes straight into out_root. The latest capture for a target number owns
    # the name (overwrites), so the file is always the most recent shot of that
    # target. `camera`/`ts` are accepted for call-compat but not used in the name.
    out_dir = Path(out_root).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    team = team or "unknown"
    fname = f"Task_2_{team}_target_{target_num}.jpg"
    path = out_dir / fname
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise IOError(f"cv2.imwrite failed for {path}")
    return path
