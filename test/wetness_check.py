"""Purple->blue wetness determination for Task-2 extinguish confirmation.

The targets are paper circles dyed in pH-sensitive red-cabbage juice (PURPLE).
The flight-line water has baking soda mixed in (slightly basic), so a wetted
target turns BLUE. Purple and blue are *adjacent* HSV hues, and the rules note
the colours persist in rain but LOSE SATURATION -- so the reliable signal is the
HUE SHIFT, not raw blue saturation.

Robustness layers (all config-driven, field-tunable on cabbage-dye samples):
  (a) region masking      -- isolate dye pixels from the white plastic backing.
  (b) two decision modes  -- shift (autonomous, self-calibrating vs a pre-fire
                             purple baseline) and absolute (manual, no baseline).
  (c) frame aggregation   -- majority vote over the N verify frames.
  (d) three-state output  -- "WETTED" | "UNCERTAIN" | "NOT_WETTED".

Only "WETTED" should advance the target counter. "UNCERTAIN"/"NOT_WETTED" defer
to the operator. The caller must ALWAYS save the photo regardless of the result
-- this module never gates the evidence, only the automatic numbering.

OpenCV HSV hue is 0..179. Purple ~140, blue ~107, so wetting moves the median
hue DOWN. ``baseline_hue - median`` is positive when a target wets.
"""

import numpy as np
import cv2

DEFAULT_CFG = {
    "purple_h": [125, 160],   # purple/magenta hue band (unwetted)
    "blue_h": [90, 125],      # blue hue band (wetted); boundary at 125
    "sat_min": 25,            # drop white backing (low S); permissive for rain desaturation
    "val_min": 30,            # drop dark pixels
    "val_max": 245,           # drop blown-out white
    "wetted_threshold": 0.5,  # absolute mode: blue must dominate purple by this fraction
    "shift_margin": 12,       # shift mode: min hue-degree drop to count as moved
    "min_colored_px": 60,     # below this many dye pixels -> UNCERTAIN
    "center_crop_frac": 0.4,  # fallback region size when no bbox is supplied
}


def _merge(cfg):
    m = dict(DEFAULT_CFG)
    if cfg:
        m.update({k: v for k, v in cfg.items() if k in DEFAULT_CFG})
    return m


def _region(frame, bbox, cfg):
    """Crop to the target bbox, or a center crop if no bbox is given."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    h, w = frame.shape[:2]
    if bbox is not None:
        (x1, y1), (x2, y2) = bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
    frac = float(cfg.get("center_crop_frac", 0.4))
    cw = max(1, int(w * frac)); ch = max(1, int(h * frac))
    cx, cy = w // 2, h // 2
    return frame[max(0, cy - ch // 2): cy + ch // 2,
                 max(0, cx - cw // 2): cx + cw // 2]


def _dye_pixels(frame, bbox, cfg):
    """Return (hue_array_int16, count) of dye-coloured pixels in the region:
    pixels that are saturated enough (not white backing), bright enough (not
    dark), and within the purple..blue hue family."""
    region = _region(frame, bbox, cfg)
    if region is None or region.size == 0:
        return np.array([], dtype=np.int16), 0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    dye_lo = min(cfg["blue_h"][0], cfg["purple_h"][0])
    dye_hi = max(cfg["blue_h"][1], cfg["purple_h"][1])
    mask = ((S >= cfg["sat_min"]) & (V >= cfg["val_min"]) & (V <= cfg["val_max"])
            & (H >= dye_lo) & (H <= dye_hi))
    hues = H[mask]
    return hues, int(hues.size)


def _ratios(hues, cfg):
    """Return (blue_ratio, purple_ratio, median_hue|None) over the dye pixels."""
    if hues.size == 0:
        return 0.0, 0.0, None
    blue = int(((hues >= cfg["blue_h"][0]) & (hues < cfg["blue_h"][1])).sum())
    purple = int(((hues >= cfg["purple_h"][0]) & (hues <= cfg["purple_h"][1])).sum())
    total = int(hues.size)
    return blue / total, purple / total, float(np.median(hues))


def baseline_hue(frame, bbox, cfg=None):
    """Median dye-pixel hue of a pre-fire (purple) target, or None if too few
    dye pixels. The autonomous caller smooths this over several frames."""
    cfg = _merge(cfg)
    hues, n = _dye_pixels(frame, bbox, cfg)
    if n < cfg["min_colored_px"]:
        return None
    return float(np.median(hues))


def assess_absolute(frame, bbox, cfg=None):
    """Manual-path determination with no baseline. Returns
    (state, blue_ratio, purple_ratio, n_px)."""
    cfg = _merge(cfg)
    hues, n = _dye_pixels(frame, bbox, cfg)
    blue_r, purple_r, med = _ratios(hues, cfg)
    if n < cfg["min_colored_px"] or med is None:
        return ("UNCERTAIN", blue_r, purple_r, n)
    in_blue = cfg["blue_h"][0] <= med < cfg["blue_h"][1]
    in_purple = cfg["purple_h"][0] <= med <= cfg["purple_h"][1]
    if in_blue and (blue_r - purple_r) >= cfg["wetted_threshold"]:
        return ("WETTED", blue_r, purple_r, n)
    if in_purple and (purple_r - blue_r) >= cfg["wetted_threshold"]:
        return ("NOT_WETTED", blue_r, purple_r, n)
    return ("UNCERTAIN", blue_r, purple_r, n)


def assess_shift(frame, bbox, baseline, cfg=None):
    """Autonomous-path determination vs a pre-fire purple ``baseline`` hue.
    Returns (state, blue_ratio, purple_ratio, n_px).

    Without a baseline we fall back to absolute classification but never
    auto-confirm WETTED (bias toward UNCERTAIN -- the operator decides)."""
    cfg = _merge(cfg)
    if baseline is None:
        state, blue_r, purple_r, n = assess_absolute(frame, bbox, cfg)
        if state == "WETTED":
            state = "UNCERTAIN"
        return (state, blue_r, purple_r, n)
    hues, n = _dye_pixels(frame, bbox, cfg)
    blue_r, purple_r, med = _ratios(hues, cfg)
    if n < cfg["min_colored_px"] or med is None:
        return ("UNCERTAIN", blue_r, purple_r, n)
    shift = baseline - med  # positive => moved purple->blue (hue dropped)
    landed_blue = med < cfg["purple_h"][0]
    if shift >= cfg["shift_margin"] and landed_blue:
        return ("WETTED", blue_r, purple_r, n)
    if med >= cfg["purple_h"][0] and shift < cfg["shift_margin"]:
        return ("NOT_WETTED", blue_r, purple_r, n)
    return ("UNCERTAIN", blue_r, purple_r, n)


def aggregate(results):
    """Combine per-frame results into one decision. ``results`` is a list of
    (state, blue_ratio, purple_ratio, n_px). Strict-majority vote; ties or a
    lack of confident frames -> UNCERTAIN. Ratios reported as medians."""
    if not results:
        return ("UNCERTAIN", 0.0, 0.0, 0)
    states = [r[0] for r in results]
    wet = states.count("WETTED")
    notwet = states.count("NOT_WETTED")
    blue = float(np.median([r[1] for r in results]))
    purple = float(np.median([r[2] for r in results]))
    npx = int(np.median([r[3] for r in results]))
    need = (len(results) // 2) + 1  # strict majority
    if wet >= need and wet > notwet:
        return ("WETTED", blue, purple, npx)
    if notwet >= need and notwet > wet:
        return ("NOT_WETTED", blue, purple, npx)
    return ("UNCERTAIN", blue, purple, npx)
