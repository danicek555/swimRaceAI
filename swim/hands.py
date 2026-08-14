"""swimRaceAI — hands."""

import base64
import json
import os
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from .config import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .blocks import *  # noqa: F401,F403
from .vlm import *  # noqa: F401,F403
from .tracking import *  # noqa: F401,F403


def _hand_box_is_plausible(
    box: tuple[int, int, int, int],
    leave_scale: float,
) -> bool:
    """Reject collapsed / splash / neighbor SAM boxes after the dive starts."""
    bw, bh = _box_size(box)
    aspect = bw / bh
    if aspect > 3.0 or aspect < 0.33:
        return False
    if min(bw, bh) < 45:
        return False
    if max(bw, bh) < 0.40 * max(leave_scale, 80.0):
        return False
    return True



def hand_follow_center(
    samples: list[tuple[float, float, float] | None] | None,
    track_boxes: list[tuple[int, int, int, int] | None],
    fps: float,
    t: float,
    leave_time: float,
    leave_center: tuple[float, float],
    leave_scale: float,
    vx: float,
    vy: float,
) -> tuple[float, float, str]:
    """
    Body center for the hand-entry crop.

    Prefer a plausible live track/sample; otherwise coast on dive velocity
    from leave (SAM often collapses onto splash / officials mid-entry).
    """
    idx = int(round(t * fps))
    idx = min(max(idx, 0), max(len(track_boxes) - 1, 0))
    box = nearest_track_box(track_boxes, idx) if track_boxes else None
    sample = None
    if samples:
        sample = nearest_frame_sample(samples, idx)

    if box is not None and _hand_box_is_plausible(box, leave_scale):
        if sample is not None:
            return float(sample[0]), float(sample[1]), "track"
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        return float(cx), float(cy), "box"

    dt = max(0.0, t - leave_time)
    return leave_center[0] + vx * dt, leave_center[1] + vy * dt, "coast"



def hand_entry_crop_rect(
    frame_shape: tuple[int, ...],
    cx: float,
    cy: float,
    dive_sign: int,
    side: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Crop centered slightly AHEAD of the body so outstretched hands stay visible.

    Wider than tall (dive is horizontal). Strong forward bias toward dive
    direction — a hip-centered square cuts the hands off the leading edge.
    """
    h, w = frame_shape[:2]
    side = int(side if side is not None else min(VLM_CROP_MIN_SIDE + 40, w, h, 440))
    side = max(280, min(side, w, h))
    crop_h = side
    crop_w = int(min(w, max(side, int(side * 1.45))))
    # Hands lead the dive: shift crop center well toward the water.
    cx = float(cx) + dive_sign * crop_w * 0.32
    cy = float(cy) + crop_h * 0.04
    xa = int(round(cx - crop_w / 2))
    ya = int(round(cy - crop_h / 2))
    xb = xa + crop_w
    yb = ya + crop_h
    if xa < 0:
        xb -= xa
        xa = 0
    if ya < 0:
        yb -= ya
        ya = 0
    if xb > w:
        xa -= xb - w
        xb = w
    if yb > h:
        ya -= yb - h
        yb = h
    xa = max(0, xa)
    ya = max(0, ya)
    xb = min(w, max(xa + 1, xb))
    yb = min(h, max(ya + 1, yb))
    return int(xa), int(ya), int(xb), int(yb)



def hand_splash_metrics(
    crop_bgr: np.ndarray,
    prev_gray: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Local splash energy: bright foam fraction + frame difference.

    Returns (white_frac, delta_0_to_1, gray).
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Entry foam is bright and desaturated.
    bright = (hsv[:, :, 2] >= 195) & (hsv[:, :, 1] <= 90)
    white_frac = float(np.mean(bright))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    delta = 0.0
    if prev_gray is not None and prev_gray.shape == gray.shape:
        delta = float(np.mean(cv2.absdiff(gray, prev_gray))) / 255.0
    return white_frac, delta, gray



def fuse_hand_label(
    vlm_label: str | None,
    white_frac: float,
    delta: float,
    src: str,
) -> tuple[str, str]:
    """
    Combine VLM + local splash. Returns (label, note).

    UNKNOWN stays UNKNOWN. WATER needs splash support when coasting.
    Strong splash can upgrade AIR → WATER.
    """
    if vlm_label is None:
        return "HANDS_UNKNOWN", "no_vlm"
    splash_hit = (
        white_frac >= VLM_HAND_SPLASH_WHITE_FRAC
        or delta >= VLM_HAND_SPLASH_DELTA
    )
    strong = white_frac >= VLM_HAND_SPLASH_STRONG_FRAC

    if vlm_label == "HANDS_UNKNOWN":
        return "HANDS_UNKNOWN", "occluded"

    if vlm_label == "HANDS_WATER":
        if splash_hit or src == "track":
            return "HANDS_WATER", "vlm+splash" if splash_hit else "vlm_track"
        # Coast WATER without foam is unreliable (hat / wrong coast).
        if src == "coast":
            return "HANDS_UNKNOWN", "coast_water_no_splash"
        return "HANDS_WATER", "vlm"

    if vlm_label == "HANDS_AIR":
        if strong:
            return "HANDS_WATER", "splash_override"
        return "HANDS_AIR", "vlm"

    return "HANDS_UNKNOWN", "fallback"



def smooth_vlm_isolated_air(
    labels: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """Ignore a single AIR blip between two WATER answers."""
    if len(labels) < 3:
        return labels
    out = list(labels)
    for i in range(1, len(out) - 1):
        if (
            out[i][1] == "HANDS_AIR"
            and out[i - 1][1] == "HANDS_WATER"
            and out[i + 1][1] == "HANDS_WATER"
        ):
            out[i] = (out[i][0], "HANDS_WATER")
    return out



def confirmed_hand_entry_time(labels: list[tuple[float, str]]) -> float | None:
    """First of VLM_HAND_CONFIRM WATER in a row after last AIR (skip UNKNOWN)."""
    filtered = [(t, lab) for t, lab in labels if lab in ("HANDS_AIR", "HANDS_WATER")]
    filtered = smooth_vlm_isolated_air(filtered)
    need = max(1, int(VLM_HAND_CONFIRM))
    last_air = -1
    for i, (_t, label) in enumerate(filtered):
        if label == "HANDS_AIR":
            last_air = i
    start = last_air + 1
    for i in range(start, len(filtered)):
        if filtered[i][1] != "HANDS_WATER":
            continue
        streak = 0
        for j in range(i, len(filtered)):
            if filtered[j][1] != "HANDS_WATER":
                break
            streak += 1
            if streak >= need:
                return float(filtered[i][0])
    return None



class HandEntryResult(NamedTuple):
    """Hand-entry timing with confidence."""

    time: float | None
    t_lo: float | None
    t_hi: float | None
    status: str  # confident | estimated | unknown
    detail: str



def estimate_hand_entry_range(
    labels: list[tuple[float, str]],
    leave_time: float,
) -> HandEntryResult:
    """Build a range when we lack a confirmed streak (often occlusion)."""
    last_air: float | None = None
    first_water: float | None = None
    n_unk = sum(1 for _, lab in labels if lab == "HANDS_UNKNOWN")
    n = max(len(labels), 1)
    for t, lab in labels:
        if lab == "HANDS_AIR":
            last_air = float(t)
    for t, lab in labels:
        # A WATER answer followed by a later AIR is DISPROVEN (hands cannot
        # leave the water again) — only WATER after the last AIR counts.
        if lab == "HANDS_WATER" and (last_air is None or t > last_air):
            first_water = float(t)
            break

    if last_air is not None and first_water is not None and first_water >= last_air:
        mid = 0.5 * (last_air + first_water)
        return HandEntryResult(
            mid,
            last_air,
            first_water,
            "estimated",
            f"bracket last AIR {last_air:.2f}s → first WATER {first_water:.2f}s",
        )

    if first_water is not None:
        return HandEntryResult(
            first_water,
            first_water,
            first_water,
            "estimated",
            f"single/partial WATER at {first_water:.2f}s (not fully confirmed)",
        )

    if n_unk / n >= 0.45:
        lo = leave_time + 0.30
        hi = leave_time + 0.70
        mid = 0.5 * (lo + hi)
        return HandEntryResult(
            mid,
            lo,
            hi,
            "estimated",
            f"view often occluded ({n_unk}/{n} frames); "
            f"flight estimate {lo - leave_time:.2f}–{hi - leave_time:.2f}s after leave",
        )

    return HandEntryResult(
        None,
        None,
        None,
        "unknown",
        "no clear hand-entry evidence",
    )



def lane_water_band(
    ref_frame: np.ndarray,
    lead_xy: tuple[float, float],
    leave_scale: float,
    vy: float,
    dive_sign: int,
    y_hint: float | None = None,
) -> tuple[int, int, str]:
    """
    (band_top, band_bottom, how): the y-strip of the swimmer's OWN lane water.

    Lane ropes are red, near-horizontal lines — a row profile of red pixels
    in the area ahead of the swimmer finds them, and the band is the gap the
    dive trajectory descends into. The neighbour's simultaneous entry lands
    in a DIFFERENT band, so restricting all measurements to this band is what
    keeps his splash out of our signal.
    """
    h, w = ref_frame.shape[:2]
    lx, ly = lead_xy
    # Typical flight ~0.35s. The dive is ballistic, so a LINEAR vy always
    # UNDERSTATES the descent — bias the predicted entry down a little, or a
    # borderline prediction lands the band one lane too high (the neighbour's).
    # y_hint (measured trajectory probe from SAM samples) beats prediction.
    y_entry = ly + vy * 0.35 + 0.12 * leave_scale
    if y_hint is not None:
        # The probe is a MEASUREMENT (tracked trajectory) — it replaces the
        # stacked-bias prediction instead of competing with it.
        y_entry = float(y_hint)
    if dive_sign > 0:
        xa = int(min(w - 2, lx + 0.2 * leave_scale))
        xb = int(min(w, lx + 3.4 * leave_scale))
    else:
        xa = int(max(0, lx - 3.4 * leave_scale))
        xb = int(max(2, lx - 0.2 * leave_scale))
    ya = int(max(0, y_entry - 1.6 * leave_scale))
    yb = int(min(h, y_entry + 1.6 * leave_scale))
    fallback = (
        int(max(0, y_entry - 0.30 * leave_scale)),
        int(min(h, y_entry + 0.55 * leave_scale)),
        "estimate (no ropes found)",
    )
    if xb - xa < 40 or yb - ya < 40:
        return fallback

    region = ref_frame[ya:yb, xa:xb]
    red = red_block_mask(region)
    row_red = red.sum(axis=1) / 255.0
    # Low per-row threshold: a rope crossing the strip at a perspective SLANT
    # puts only a fraction of its beads on any single row. A real rope is
    # confirmed by the GROUP total instead (summed red ~ strip width).
    rope_rows = np.where(row_red > 0.10 * (xb - xa))[0]
    if rope_rows.size == 0:
        return fallback

    # Group consecutive rope rows; keep groups whose total red area says
    # "this line crosses the whole strip".
    ropes: list[float] = []
    start = prev = int(rope_rows[0])

    def flush(a: int, b: int) -> None:
        total = float(row_red[a : b + 1].sum())
        if total >= 0.9 * (xb - xa):
            ropes.append(ya + 0.5 * (a + b))

    for r in rope_rows[1:]:
        r = int(r)
        if r - prev > 6:
            flush(start, prev)
            start = r
        prev = r
    flush(start, prev)
    if not ropes:
        return fallback

    above = [r for r in ropes if r < y_entry]
    below = [r for r in ropes if r >= y_entry]
    top = int(above[-1] + 6) if above else int(max(0, y_entry - 0.30 * leave_scale))
    bot = int(below[0] - 4) if below else int(min(h, y_entry + 0.55 * leave_scale))
    if bot - top < 18:
        return fallback
    return top, bot, f"ropes at {[int(r) for r in ropes]}"



def hand_entry_local_scan(
    capture: cv2.VideoCapture,
    fps: float,
    leave_time: float,
    end_t: float,
    lead_xy: tuple[float, float],
    leave_scale: float,
    vx: float,
    vy: float,
    dive_sign: int,
    crop_dir: Path | None = None,
    y_hint: float | None = None,
) -> tuple[float | None, tuple[int, int] | None, str]:
    """
    Frame-accurate hand-entry candidate WITHOUT the vision LM.

    Inside the swimmer's own lane band, per-frame differences are aggregated
    on a fixed cell grid. Ambient shimmer / wakes fill many cells thinly and
    flicker; the entry blob is DENSE in one cell, ABOVE that cell's ambient
    baseline, PERSISTENT at one spot, and NEAR the predicted hand path.
    Returns (time, band, note).
    """
    ref_idx = int(round(leave_time * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, ref_frame = capture.read()
    if not ok:
        return None, None, "no reference frame"

    band_top, band_bot, band_how = lane_water_band(
        ref_frame, lead_xy, leave_scale, vy, dive_sign, y_hint=y_hint
    )
    # Rope-derived bands can span several lanes when a rope goes undetected —
    # clip to about one body-scale around the predicted entry depth.
    y_entry = lead_xy[1] + vy * 0.35 + 0.12 * leave_scale
    if y_hint is not None:
        y_entry = float(y_hint)
    max_band = int(1.0 * leave_scale)
    if band_bot - band_top > max_band:
        band_top = int(max(band_top, y_entry - 0.45 * max_band))
        band_bot = int(min(band_bot, band_top + max_band))
    h, w = ref_frame.shape[:2]
    band_top = max(0, min(band_top, h - 20))
    band_bot = max(band_top + 12, min(band_bot, h))

    # Fixed x-strip covering the whole flight reach (fixed coords keep the
    # cell grid aligned across frames, so per-cell baselines make sense).
    if dive_sign > 0:
        x0 = int(max(0, lead_xy[0] - 0.2 * leave_scale))
        x1 = int(min(w, lead_xy[0] + 3.6 * leave_scale))
    else:
        x0 = int(max(0, lead_xy[0] - 3.6 * leave_scale))
        x1 = int(min(w, lead_xy[0] + 0.2 * leave_scale))
    if x1 - x0 < 60:
        return None, (band_top, band_bot), "strip too narrow"

    # Keep only actual WATER columns: on a wide reach the strip otherwise
    # runs past the pool edge onto the deck (officials, camera crew — their
    # white shirts pass the foam gate and poison cells and baselines).
    ref_strip0 = ref_frame[band_top:band_bot, x0:x1]
    hsv0 = cv2.cvtColor(ref_strip0, cv2.COLOR_BGR2HSV)
    water_col = (
        (hsv0[:, :, 0] >= 75)
        & (hsv0[:, :, 0] <= 110)
        & (hsv0[:, :, 1] >= 40)
    ).mean(axis=0)
    is_water = water_col >= 0.45
    start_col = int(
        np.clip(
            lead_xy[0] + dive_sign * 0.3 * leave_scale - x0,
            0,
            len(is_water) - 1,
        )
    )
    lo = hi = start_col
    gap = 0
    while lo > 0 and gap <= 12:
        gap = gap + 1 if not is_water[lo - 1] else 0
        lo -= 1
    gap = 0
    while hi < len(is_water) - 1 and gap <= 12:
        gap = gap + 1 if not is_water[hi + 1] else 0
        hi += 1
    if hi - lo >= 60:
        x0, x1 = x0 + lo, x0 + hi + 1

    cell = max(12, int(HAND_LOCAL_CELL_PX))
    gh = max(1, (band_bot - band_top) // cell)
    gw = max(1, (x1 - x0) // cell)
    sh, sw = gh * cell, gw * cell

    def cell_density(prev_bgr: np.ndarray, cur_bgr: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(cur_bgr, prev_bgr).max(axis=2)
        hsv = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2HSV)
        foam = (hsv[:, :, 2] >= 190) & (hsv[:, :, 1] <= 95)
        ripple = (
            (hsv[:, :, 0] >= 75)
            & (hsv[:, :, 0] <= 110)
            & (hsv[:, :, 1] >= 40)
            & (hsv[:, :, 2] >= 60)
        )
        disturb = (diff > 24) & (foam | ripple)
        d = disturb[:sh, :sw].astype(np.float32)
        return d.reshape(gh, cell, gw, cell).mean(axis=(1, 3))

    # Pass 1: per-frame cell densities from just after leave to end_t.
    first_idx = int(round((leave_time + 0.03) * fps))
    last_idx = int(round(end_t * fps))
    if last_idx <= first_idx + 5:
        return None, (band_top, band_bot), "window empty"
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
    prev_strip: np.ndarray | None = None
    dmaps: list[tuple[float, np.ndarray, np.ndarray]] = []
    for fi in range(first_idx, last_idx + 1):
        ok, frame = capture.read()
        if not ok:
            break
        strip = frame[band_top:band_bot, x0:x1]
        if prev_strip is not None:
            dmaps.append((fi / fps, cell_density(prev_strip, strip), strip.copy()))
        prev_strip = strip
    if len(dmaps) < 6:
        return None, (band_top, band_bot), "too few frames"

    # Ambient baseline per cell: the first few frames are still pure flight.
    n_base = min(4, max(2, len(dmaps) // 6))
    baseline = np.median(np.stack([d for _t, d, _s in dmaps[:n_base]]), axis=0)

    min_t = leave_time + HAND_LOCAL_MIN_FLIGHT_SECONDS
    hits = 0
    entry_t: float | None = None
    prev_cell: tuple[int, int] | None = None
    best_note = ""
    peak = 0.0
    for t, dmap, strip in dmaps:
        signal = dmap - baseline
        # Eligible cells: a GROWING CONE ahead of the leave point instead of
        # a window glued to the predicted hand position. The cone's far edge
        # advances with a generous speed cap, so a dead SAM track (bad vx)
        # cannot park the attention behind the swimmer — while early frames
        # still exclude far cells (the neighbour's early splash bleeding
        # over the rope stays out).
        v_max = max(1.6 * abs(vx), 3.5 * leave_scale)
        reach = 0.15 * leave_scale + v_max * max(0.0, t - leave_time)
        centers = x0 + (np.arange(gw) + 0.5) * cell
        if dive_sign > 0:
            eligible = (centers >= lead_xy[0] + 0.15 * leave_scale) & (
                centers <= lead_xy[0] + reach
            )
        else:
            eligible = (centers <= lead_xy[0] - 0.15 * leave_scale) & (
                centers >= lead_xy[0] - reach
            )
        if not np.any(eligible):
            continue
        masked = signal[:, eligible]
        s = float(masked.max())
        peak = max(peak, s)
        gy, gxm = np.unravel_index(int(masked.argmax()), masked.shape)
        gx = int(np.where(eligible)[0][gxm])
        if crop_dir is not None:
            vis = strip.copy()
            cv2.rectangle(
                vis,
                (gx * cell, gy * cell),
                (gx * cell + cell, gy * cell + cell),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                vis, f"t={t:.2f} S={s:.2f}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.imwrite(str(crop_dir / f"local_t{t:05.2f}s.jpg"), vis)
        if t < min_t:
            continue
        # The blob may drift with the flying hands (~0.7 cell/frame) and the
        # argmax hops between the hand cell and the growing splash — allow
        # 2 cells of drift, and a far jump RESTARTS the run instead of
        # zeroing it (the jump target may be the true entry).
        same_spot = (
            prev_cell is not None
            and abs(prev_cell[0] - gy) <= 2
            and abs(prev_cell[1] - gx) <= 2
        )
        if s >= HAND_LOCAL_CELL_SIGNAL:
            if hits == 0 or same_spot:
                hits += 1
                if entry_t is None:
                    entry_t = float(t)
            else:
                hits = 1
                entry_t = float(t)
            prev_cell = (gy, gx)
            if hits >= HAND_LOCAL_CONFIRM_FRAMES:
                note = (
                    f"band y {band_top}-{band_bot} ({band_how}); "
                    f"cell {cell}px, signal {s:.2f} at grid ({gy},{gx})"
                )
                return entry_t, (band_top, band_bot), note
        else:
            hits = 0
            entry_t = None
            prev_cell = (gy, gx) if s >= 0.5 * HAND_LOCAL_CELL_SIGNAL else None
    return None, (band_top, band_bot), (
        f"no persistent blob (peak {peak:.2f} < {HAND_LOCAL_CELL_SIGNAL}); "
        f"band y {band_top}-{band_bot} ({band_how})"
    )



def find_hand_entry_with_vlm(
    clip_path: Path,
    track_boxes: list[tuple[int, int, int, int] | None],
    fps: float,
    leave_time: float,
    samples: list[tuple[float, float, float] | None] | None = None,
    beep_time: float | None = None,
    crop_dir: Path | None = None,
) -> HandEntryResult:
    """
    First hand-water contact after leave-block.

    Order of evidence:
      1. Local scan (free, every frame): surface disturbance in the swimmer's
         own lane band vs a clean reference — frame-accurate and immune to
         the neighbour's simultaneous splash. Works without an API key.
      2. Vision LM confirms 2-3 crops around the local candidate.
      3. Full LM sweep only as fallback (occlusions, no local signal),
         with splash fusion; occluded frames are UNKNOWN. If confirmation
         fails, returns an estimated range.
    """
    if fps <= 0 or not track_boxes:
        return HandEntryResult(None, None, None, "unknown", "no track")

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        print("Could not open clip for hand-entry crops.")
        return HandEntryResult(None, None, None, "unknown", "cannot open clip")

    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
        for old in crop_dir.glob("*.jpg"):
            old.unlink()

    end_t = min(
        leave_time + VLM_HAND_SEARCH_SECONDS,
        (len(track_boxes) - 1) / fps,
    )
    start_t = leave_time
    if start_t >= end_t:
        print(
            f"Hand-entry LM: search window empty "
            f"(start {start_t:.2f}s >= end {end_t:.2f}s)."
        )
        capture.release()
        return HandEntryResult(None, None, None, "unknown", "empty window")

    dive_sign = 1
    if samples is not None and beep_time is not None:
        dive_sign = dive_direction_sign(samples, fps, beep_time)

    labels: list[tuple[float, str]] = []
    leave_idx = int(round(leave_time * fps))
    seed_box = nearest_track_box(track_boxes, leave_idx)
    if seed_box is None:
        seed_box = nearest_track_box(track_boxes, max(0, leave_idx - 3))
    leave_sample = nearest_frame_sample(samples, leave_idx) if samples else None
    if leave_sample is not None:
        leave_center = (float(leave_sample[0]), float(leave_sample[1]))
    elif seed_box is not None:
        leave_center = (
            0.5 * (seed_box[0] + seed_box[2]),
            0.5 * (seed_box[1] + seed_box[3]),
        )
    else:
        leave_center = (0.0, 0.0)
    leave_scale = 200.0
    if seed_box is not None:
        leave_scale = float(
            max(seed_box[2] - seed_box[0], seed_box[3] - seed_box[1], 80)
        )
    vx, vy = 0.0, 0.0
    if samples is not None:
        vx, vy = dive_velocity_px(samples, fps, leave_time)
    if abs(vx) < 40.0:
        vx = dive_sign * max(220.0, 1.1 * leave_scale)
    if abs(vy) < 20.0:
        vy = max(80.0, 0.35 * leave_scale)

    crop_side = int(np.clip(1.70 * leave_scale, 360, 520))

    # ---- Stage 1: local scan (free, frame-accurate, band-restricted) ----
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
    lead_x = leave_center[0]
    if seed_box is not None:
        lead_x = float(seed_box[2] if dive_sign >= 0 else seed_box[0])
    # Hands fly BELOW the body center at leave (arms swung down-forward).
    lead_y = leave_center[1] + 0.18 * leave_scale
    # Measured trajectory probe: how deep the tracked body actually got
    # during the flight — beats any prediction for picking the lane band.
    y_probe: float | None = None
    if samples:
        lo = int(round(leave_time * fps))
        # Only up to the TYPICAL flight end — later frames have the torso
        # already underwater and the sunken mask center drags the probe a
        # band too deep.
        hi = min(int(round((leave_time + 0.32) * fps)), len(samples) - 1)
        depths = [
            s[1] + 0.08 * leave_scale
            for s in samples[lo : hi + 1]
            if s is not None
        ]
        if depths:
            # Median-of-deepest: one bad SAM frame must not pick the band.
            depths.sort()
            tail = depths[-3:] if len(depths) >= 3 else depths
            y_probe = float(np.median(tail))
    local_t, band, local_note = hand_entry_local_scan(
        capture,
        fps,
        leave_time,
        end_t,
        (lead_x, lead_y),
        leave_scale,
        vx,
        vy,
        dive_sign,
        crop_dir=crop_dir,
        y_hint=y_probe,
    )
    if local_t is not None:
        print(f"Hand-entry local scan: candidate {local_t:.2f}s ({local_note})")
        if vlm_api_key() is None:
            capture.release()
            return HandEntryResult(
                local_t,
                local_t - 1.0 / fps,
                local_t + 1.0 / fps,
                "confident",
                "local splash scan (no LM available)",
            )
        # ---- Stage 2: LM confirms 3 crops around the candidate ----
        confirm_labels: list[tuple[float, str]] = []
        for tc in (local_t - 0.08, local_t + 0.05, local_t + 0.15):
            tc = float(np.clip(tc, leave_time, end_t))
            fi = min(max(int(round(tc * fps)), 0), len(track_boxes) - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = capture.read()
            if not ok:
                continue
            cx, cy, src = hand_follow_center(
                samples, track_boxes, fps, tc, leave_time,
                leave_center, leave_scale, vx, vy,
            )
            rect = hand_entry_crop_rect(frame.shape, cx, cy, dive_sign, side=crop_side)
            crop = extract_crop(frame, rect)
            white_frac, delta, _gray = hand_splash_metrics(crop)
            try:
                vlm_label, _raw = ask_vlm_hands_air_or_water(crop)
            except RuntimeError as err:
                if str(err) == "vlm_auth":
                    capture.release()
                    return HandEntryResult(
                        local_t,
                        local_t - 1.0 / fps,
                        local_t + 1.0 / fps,
                        "confident",
                        "local splash scan (LM auth failed)",
                    )
                raise
            label, _note = fuse_hand_label(vlm_label, white_frac, delta, src)
            print(f"  confirm t={tc:.2f}s  {label}")
            confirm_labels.append((tc, label))
        post = [lab for tc, lab in confirm_labels if tc > local_t]
        contradicted = len(post) >= 2 and all(lab == "HANDS_AIR" for lab in post)
        if not contradicted:
            capture.release()
            return HandEntryResult(
                local_t,
                local_t - 1.0 / fps,
                local_t + 1.0 / fps,
                "confident",
                "local splash scan + LM confirm",
            )
        print(
            "Hand-entry: LM contradicts the local candidate "
            "(AIR after it) — falling back to the full LM sweep."
        )
    else:
        print(f"Hand-entry local scan: no candidate ({local_note})")
        if vlm_api_key() is None:
            capture.release()
            return HandEntryResult(
                None, None, None, "unknown", "no API key; local scan empty"
            )

    # ---- Stage 3: full LM sweep (fallback) ----
    sample_times = np.arange(start_t, end_t + 1e-9, VLM_HAND_SAMPLE_STEP_SECONDS)
    print(
        f"Hand-entry LM ({VLM_MODEL}): checking {len(sample_times)} crops "
        f"every {VLM_HAND_SAMPLE_STEP_SECONDS:.2f}s "
        f"from {start_t:.2f}s to {end_t:.2f}s "
        f"(need {VLM_HAND_CONFIRM} WATER; splash vote + occluded=UNKNOWN)..."
    )
    if crop_dir is not None:
        print(f"Saving hand-entry crops in: {crop_dir}")
    print(
        f"Hand-entry crops: ~{int(crop_side * 1.45)}x{crop_side}px "
        f"shifted toward hands (dive {'right' if dive_sign > 0 else 'left'}; "
        f"vel≈({vx:.0f},{vy:.0f})px/s)"
    )

    prev_gray: np.ndarray | None = None
    upgraded_water_times: set[float] = set()
    for t in sample_times:
        frame_index = int(round(t * fps))
        frame_index = min(max(frame_index, 0), len(track_boxes) - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        cx, cy, src = hand_follow_center(
            samples,
            track_boxes,
            fps,
            float(t),
            leave_time,
            leave_center,
            leave_scale,
            vx,
            vy,
        )
        box = nearest_track_box(track_boxes, frame_index)
        if box is not None and _hand_box_is_plausible(box, leave_scale):
            lead_x = float(box[2] if dive_sign >= 0 else box[0])
            cx = 0.45 * cx + 0.55 * lead_x
        rect = hand_entry_crop_rect(frame.shape, cx, cy, dive_sign, side=crop_side)
        crop = extract_crop(frame, rect)
        white_frac, delta, gray = hand_splash_metrics(crop, prev_gray)
        prev_gray = gray
        try:
            vlm_label, raw_vlm = ask_vlm_hands_air_or_water(crop)
        except RuntimeError as err:
            if str(err) == "vlm_auth":
                capture.release()
                print("Hand-entry LM: HTTP 401 — skipping hand entry.")
                return HandEntryResult(None, None, None, "unknown", "auth")
            raise
        label, fuse_note = fuse_hand_label(vlm_label, white_frac, delta, src)
        # Physics guards: WATER cannot happen before the minimum flight, and
        # long after leave the hands ARE in the water even when hidden.
        if (
            label == "HANDS_WATER"
            and t < leave_time + HAND_LOCAL_MIN_FLIGHT_SECONDS
        ):
            label = "HANDS_UNKNOWN"
            fuse_note += "+too_early_for_water"
        elif (
            label == "HANDS_UNKNOWN"
            and t >= leave_time + HAND_UNKNOWN_IS_WATER_AFTER_SECONDS
        ):
            label = "HANDS_WATER"
            upgraded_water_times.add(float(t))
            fuse_note += "+late_unknown=water"
        detail = (
            f"{src} c=({cx:.0f},{cy:.0f}) "
            f"splash={white_frac:.2f}/{delta:.2f} {fuse_note}"
        )
        if raw_vlm:
            one_line = " ".join(raw_vlm.split())
            if len(one_line) > 90:
                one_line = one_line[:87] + "..."
            detail += f" | {one_line}"
        tag = label if label is not None else "NO_ANSWER"
        if crop_dir is not None:
            vis = crop.copy()
            cv2.putText(
                vis,
                detail,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            ch, cw = vis.shape[:2]
            cv2.drawMarker(
                vis,
                (cw // 2, ch // 2),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=1,
            )
            out_name = f"t{t:05.2f}s_{tag}.jpg".replace(":", "-")
            cv2.imwrite(str(crop_dir / out_name), vis)
        if label is None:
            print(f"  t={t:.2f}s  ?  ({detail})")
            continue
        # Sticky WATER only for non-occluded coast after a real WATER.
        if (
            label == "HANDS_AIR"
            and any(lab == "HANDS_WATER" for _, lab in labels)
            and src == "coast"
        ):
            label = "HANDS_WATER"
            detail += " | sticky_WATER"
            tag = label
        print(f"  t={t:.2f}s  {label}  ({detail})")
        labels.append((float(t), label))

        entry = confirmed_hand_entry_time(labels)
        if entry is not None:
            water_count = sum(1 for _, lab in labels if lab == "HANDS_WATER")
            if water_count >= VLM_HAND_CONFIRM + 1:
                break

    capture.release()
    entry = confirmed_hand_entry_time(labels)
    if entry is not None:
        if entry in upgraded_water_times:
            # The streak was carried by late-UNKNOWN upgrades — that gives an
            # UPPER BOUND ("no later than"), not a measurement. Return the
            # honest bracket last AIR -> first upgraded WATER instead.
            airs = [t for t, lab in labels if lab == "HANDS_AIR" and t < entry]
            lo = max(airs) if airs else leave_time + HAND_LOCAL_MIN_FLIGHT_SECONDS
            mid = 0.5 * (lo + entry)
            print(
                f"Hand-entry estimated {lo:.2f}-{entry:.2f}s "
                f"(streak confirmed only by late UNKNOWN=water upgrades)"
            )
            return HandEntryResult(
                mid, lo, entry, "estimated",
                "bracket last AIR -> late hidden entry",
            )
        print(
            f"Hand-entry confident at {entry:.2f}s "
            f"({VLM_HAND_CONFIRM} WATER in a row after last AIR)"
        )
        return HandEntryResult(
            entry, entry, entry, "confident", "confirmed WATER streak"
        )

    result = estimate_hand_entry_range(labels, leave_time)
    if result.status == "estimated":
        print(
            f"Hand-entry ESTIMATED {result.t_lo:.2f}–{result.t_hi:.2f}s "
            f"(~{result.time:.2f}s) — {result.detail}"
        )
        print(
            "  Not exact: occlusion and/or missing confirmed streak. "
            "Treat as a range, not a pad-accurate time."
        )
    else:
        print(f"Hand-entry unknown — {result.detail}")
    return result




