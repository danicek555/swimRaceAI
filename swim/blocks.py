"""swimRaceAI — blocks."""

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


# --- NEPOUZIVANE (stinovana prvni verze red_block_mask — runtime vzdy pouzival druhou definici; zachovano pro referenci) — ponechano zakomentovane pro pripadne oziveni ---
# def red_block_mask(frame: np.ndarray) -> np.ndarray:
#     """White pixels = reddish starting-block color. HSV handles lighting better than RGB."""
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # Red wraps around hue 0, so we need two ranges.
#     low_red = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([15, 255, 255]))
#     high_red = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
#     mask = cv2.bitwise_or(low_red, high_red)
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     # Starting blocks are on the LEFT. Red lane lines in the water are on the RIGHT.
#     # Zero the right side so people in the water do not count as "on a block".
#     cutoff = int(mask.shape[1] * 0.55)
#     mask[:, cutoff:] = 0
#     return mask
#
#
# --- konec nepouzivaneho bloku ---

def red_block_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of red starting-block tops in a BGR image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 60, 60), (14, 255, 255))
    mask2 = cv2.inRange(hsv, (165, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(mask1, mask2)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return red



def primary_block_contour(
    red: np.ndarray,
    anchor_xy: tuple[float, float] | None = None,
    max_top_below: float | None = None,
    min_area_frac: float = 0.006,
) -> np.ndarray | None:
    """
    Pick ONE red platform contour (not lane ropes / skin blobs).

    The anchor is the swimmer's FEET. Physical prior: the supporting platform
    has its TOP EDGE at foot level — so score by top-edge distance, never by
    raw area. Area-based scoring picked the NEIGHBOUR's platform whenever the
    swimmer's own (smaller, half-hidden under their body) platform lost the
    size contest — every lane except the nearest one.

    max_top_below: reject candidates whose top edge lies more than this many
    pixels BELOW the feet — a platform far under the feet cannot be the one
    they stand on (that is the nearer lane's block).
    """
    h, w = red.shape[:2]
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(120.0, min_area_frac * h * w)
    best = None
    best_penalty = 1e18
    ax = float(anchor_xy[0]) if anchor_xy else w * 0.35
    ay = float(anchor_xy[1]) if anchor_xy else h * 0.65
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Reject thin lane-rope strips and tiny blobs.
        if bh < 10 and bw > 3 * max(bh, 1):
            continue
        if bw < 18 or bh < 10:
            continue
        cx = x + bw * 0.5
        # Prefer left half (blocks), not mid-pool.
        if cx > 0.72 * w:
            continue
        if max_top_below is not None and (y - ay) > max_top_below:
            continue
        # Feet stand at the platform's FRONT edge, not its center — measure
        # x as distance to the contour's x-RANGE (0 when the anchor is above
        # the platform), otherwise a small front-edge fragment near the toes
        # outscores the full platform whose center sits behind the feet.
        x_dist = max(x - ax, ax - (x + bw), 0.0)
        penalty = 2.0 * abs(y - ay) + 0.8 * x_dist
        if penalty < best_penalty:
            best_penalty = penalty
            best = cnt
    return best



def foot_skin_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Rough mask of skin / foot pixels.

    Important: do NOT include pure-red hues — those are the starting block,
    and counting them as 'skin' made empty blocks look like huge overlap.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Typical skin (avoid hue near 0 / 180 = red block tops).
    skin = cv2.inRange(hsv, (5, 40, 70), (25, 160, 255))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # Explicitly remove red-block pixels.
    skin = cv2.bitwise_and(skin, cv2.bitwise_not(red_block_mask(bgr)))
    return skin



def foot_block_overlap_px(
    bgr: np.ndarray,
    anchor_xy: tuple[float, float] | None = None,
    locked_cnt: np.ndarray | None = None,
) -> tuple[int, np.ndarray, np.ndarray | None]:
    """
    Count skin/foot pixels on the BLOCK edge / just above it.

    Returns (overlap_pixels, red_mask, primary_contour_or_None).
    Empty red platform alone must score ~0 (no foot contact).

    locked_cnt: block contour found once at beep time (crop coords). The block
    is static, so reusing it beats re-guessing per frame — and it is guaranteed
    to be the SWIMMER'S block, not a neighbouring lane's.
    """
    red = red_block_mask(bgr)
    cnt = locked_cnt if locked_cnt is not None else primary_block_contour(red, anchor_xy=anchor_xy)
    if cnt is None:
        return 0, red, None

    h, w = red.shape[:2]
    x, y, bw, bh = cv2.boundingRect(cnt)
    # Contact zone: band just above the platform + thin edge of the block.
    zone = np.zeros((h, w), dtype=np.uint8)
    band_top = max(0, y - max(12, int(0.9 * bh)))
    band_bot = min(h, y + max(8, int(0.35 * bh)))
    zone[band_top:band_bot, max(0, x - 4) : min(w, x + bw + 4)] = 255
    edge = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge, [cnt], -1, 255, thickness=5)
    contact_zone = cv2.bitwise_or(zone, edge)

    skin = foot_skin_mask(bgr)
    overlap = cv2.bitwise_and(skin, contact_zone)
    # Never count red platform fill as contact.
    overlap = cv2.bitwise_and(overlap, cv2.bitwise_not(red))
    return int(np.count_nonzero(overlap)), red, cnt



def rect_from_block_bbox(
    frame_hw: tuple[int, int],
    block_bbox: tuple[int, int, int, int],
    body_scale: float,
    dive_sign: int,
) -> tuple[int, int, int, int] | None:
    """
    Crop window from a block bbox (frame coords) + the standard pads.

    unit has a body-scale floor: at beep the visible contour is often just a
    sliver of the platform peeking out around the swimmer's body — a window
    sized from the sliver alone is unusably small (a 168x175px crop upscaled
    to 512px is pixel mush).
    """
    h, w = frame_hw[:2]
    bx, by, bw, bh = block_bbox
    unit = max(bw, bh, int(0.35 * body_scale))
    pad_fwd = int(VLM_BLOCK_PAD_FWD * unit)
    pad_back = int(VLM_BLOCK_PAD_BACK * unit)
    if dive_sign > 0:
        xa = bx - pad_back
        xb = bx + bw + pad_fwd
    else:
        xa = bx - pad_fwd
        xb = bx + bw + pad_back
    ya = by - int(VLM_BLOCK_PAD_UP * unit)
    yb = by + bh + int(VLM_BLOCK_PAD_DOWN * unit)

    # Absolute size floor. All the maths above is RELATIVE (block px, body
    # px) — on a far lane the block is tiny, the window comes out ~150px and
    # upscales to unreadable mush. A far lane needs a minimum absolute window;
    # the neighbouring lanes it pulls in are handled by dim + prompt + veto.
    # Extra space goes mostly forward (flight path) and up (legs), matching
    # the pad philosophy.
    min_side = min(VLM_CROP_MIN_SIDE, w, h)
    if xb - xa < min_side:
        extra = min_side - (xb - xa)
        if dive_sign > 0:
            xa -= int(extra * 0.30)
            xb += extra - int(extra * 0.30)
        else:
            xa -= extra - int(extra * 0.30)
            xb += int(extra * 0.30)
    if yb - ya < min_side:
        extra = min_side - (yb - ya)
        ya -= int(extra * 0.65)
        yb += extra - int(extra * 0.65)

    # Clamp with shift, so the floor survives at frame edges.
    if xa < 0:
        xb -= xa
        xa = 0
    if ya < 0:
        yb -= ya
        ya = 0
    if xb > w:
        xa = max(0, xa - (xb - w))
        xb = w
    if yb > h:
        ya = max(0, ya - (yb - h))
        yb = h
    if xb - xa < 64 or yb - ya < 64:
        return None
    return int(xa), int(ya), int(xb), int(yb)



def tight_block_crop_rect(
    frame: np.ndarray,
    foot_xy: tuple[float, float],
    body_scale: float,
    dive_sign: int,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None, bool]:
    """
    Crop rect hugging the swimmer's OWN starting block.

    The block is found ONCE, at beep time, in a small window around the feet —
    at that moment the feet are provably standing on it, so the nearest red
    platform IS the right one. Neighbouring lanes never win the search because
    they are a full lane pitch away. Pools without red blocks get a rectangle
    pseudo-contour synthesized from the feet + body size instead.

    Returns (rect, block_contour_in_crop_coords, contour_from_red).
    rect is None only when the window itself is degenerate (caller falls back
    to the body-box crop).
    """
    h, w = frame.shape[:2]
    fx, fy = int(foot_xy[0]), int(foot_xy[1])
    if not (0 <= fx < w and 0 <= fy < h):
        return None, None, False

    # Small search window around the feet. The supporting platform's TOP is
    # at foot level — reach less far down, so the nearer lane's (lower,
    # bigger) block does not even enter the contest.
    side = int(np.clip(2.4 * body_scale, 200, VLM_BLOCK_SEARCH_FRAC * 2 * min(h, w)))
    wx1 = max(0, fx - side // 2)
    wx2 = min(w, fx + side // 2)
    wy1 = max(0, fy - side // 2)
    wy2 = min(h, fy + int(side * 0.35))
    window = frame[wy1:wy2, wx1:wx2]
    if window.size == 0:
        return None, None, False

    # Low min-area: at beep the swimmer's own body hides most of their
    # platform — the visible sliver must stay a candidate. Dilation first:
    # the body splits the platform into fragments; merged they score as one
    # platform with the true top edge. Neighbouring lanes' blocks are a full
    # lane pitch away, far beyond the dilation reach.
    red = cv2.dilate(red_block_mask(window), np.ones((11, 11), np.uint8))
    cnt = primary_block_contour(
        red,
        anchor_xy=(fx - wx1, fy - wy1),
        max_top_below=0.5 * body_scale,
        min_area_frac=0.002,
    )
    if cnt is None:
        # No red found (non-red blocks / odd lighting): synthesize the block
        # from the feet + body size. Less precise, still the right lane.
        # A rectangle pseudo-contour keeps dimming + occlusion working.
        bw = int(0.95 * body_scale)
        bh = int(0.55 * body_scale)
        bx = fx - bw // 2
        by = fy - int(0.15 * bh)
        cnt_frame = np.array(
            [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
    else:
        x, y, cw, ch = cv2.boundingRect(cnt)
        bx, by, bw, bh = x + wx1, y + wy1, cw, ch
        cnt_frame = cnt + np.array([wx1, wy1], dtype=np.int32)

    rect = rect_from_block_bbox((h, w), (bx, by, bw, bh), body_scale, dive_sign)
    if rect is None:
        return None, None, False

    cnt_crop = cnt_frame - np.array([rect[0], rect[1]], dtype=np.int32)
    return rect, cnt_crop, cnt is not None



def empty_platform_contour(
    capture: cv2.VideoCapture,
    rect: tuple[int, int, int, int],
    anchor_crop: tuple[float, float] | None,
    ref_idx: int,
    beep_cnt: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    (contour, reference_frame) — contour from a frame AFTER the swimmer left.

    At beep time the feet occlude part of the red top, so the beep-time
    contour has a bite where the feet were. On the empty platform the full
    red is visible — that shape is the correct occlusion baseline. The
    reference crop itself is the baseline for colour-free frame differencing.

    Keeps beep_cnt when the reference contour looks wrong (camera pan,
    much smaller area, nothing found) — the ref FRAME is still returned so
    the caller can build diff baselines for whatever rect it settles on.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, frame = capture.read()
    if not ok:
        return beep_cnt, None
    crop = extract_crop(frame, rect)
    cnt = primary_block_contour(
        red_block_mask(crop), anchor_xy=anchor_crop, min_area_frac=0.002
    )
    if cnt is None:
        return beep_cnt, frame
    if beep_cnt is not None:
        area_new = cv2.contourArea(cnt)
        area_old = cv2.contourArea(beep_cnt)
        x0, y0, w0, h0 = cv2.boundingRect(beep_cnt)
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        center_shift = (
            ((x1 + w1 / 2) - (x0 + w0 / 2)) ** 2
            + ((y1 + h1 / 2) - (y0 + h0 / 2)) ** 2
        ) ** 0.5
        if area_new < 0.5 * area_old or center_shift > 0.35 * max(w0, h0):
            return beep_cnt, frame
    return cnt, frame



def filled_platform_mask(
    cnt: np.ndarray | None,
    shape_hw: tuple[int, int],
) -> np.ndarray | None:
    """Filled platform polygon, eroded a little so red-edge flicker never counts."""
    if cnt is None:
        return None
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.drawContours(mask, [cnt.astype(np.int32)], -1, 255, thickness=-1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    if np.count_nonzero(mask) < 200:
        return None
    return mask



def platform_occlusion(
    crop_bgr: np.ndarray,
    platform_mask: np.ndarray,
    ref_crop: np.ndarray | None = None,
) -> tuple[int, float]:
    """
    (pixels, fraction) of the platform polygon occluded by something.

    Preferred: frame differencing against the empty-platform reference crop —
    colour-free, so it works on ANY block colour and on shadowed feet.
    Fallback (no reference): non-red pixels inside the polygon.
    """
    area = max(int(np.count_nonzero(platform_mask)), 1)
    if ref_crop is not None and ref_crop.shape == crop_bgr.shape:
        now = crop_bgr.astype(np.float32)
        ref = ref_crop.astype(np.float32)
        diff = np.abs(now - ref).max(axis=2)
        changed = (diff > 28) & (platform_mask > 0)
        # Shadow rejection: a hovering foot casts a shadow on the platform —
        # a uniform darkening of the SAME colour. Real occlusion (skin over
        # the block) changes the colour ratio per channel. Without this, a
        # sunny outdoor pool reads "shadow" as "contact" one frame too long.
        ratio = now / (ref + 1.0)
        darkening = np.all((ratio > 0.35) & (ratio < 0.92), axis=2)
        uniform = (ratio.max(axis=2) - ratio.min(axis=2)) < 0.18
        occl = changed & ~(darkening & uniform)
        px = int(np.count_nonzero(occl))
        return px, px / area
    red = red_block_mask(crop_bgr)
    occl = cv2.bitwise_and(platform_mask, cv2.bitwise_not(red))
    px = int(np.count_nonzero(occl))
    return px, px / area



def scene_looks_static(
    crop_a: np.ndarray | None,
    crop_b: np.ndarray | None,
    exclude_mask: np.ndarray,
    top_limit_y: int,
) -> bool:
    """
    True when the deck above the platform barely changed between two moments
    ~1.5s apart. A panning/zooming camera fails this — then the locked window
    and the occlusion baseline are meaningless and the veto must switch off.

    Only rows above the platform top are compared: that is deck/blocks (static
    in a side view), never open water, so waves cannot false-trip the check.
    """
    if crop_a is None or crop_b is None or crop_a.shape != crop_b.shape:
        return False
    diff = cv2.absdiff(crop_a, crop_b).max(axis=2)
    zone = np.zeros(diff.shape, dtype=bool)
    zone[: max(top_limit_y, 0), :] = True
    zone &= exclude_mask == 0
    n = int(np.count_nonzero(zone))
    if n < 500:
        return True  # nothing left to judge; do not veto the veto
    changed = int(np.count_nonzero((diff > 30) & zone))
    return changed / n < 0.5



def dim_crop_outside_target(
    crop: np.ndarray,
    block_cnt: np.ndarray | None,
    swimmer_rect: tuple[int, int, int, int] | None,
    factor: float = VLM_DIM_OUTSIDE,
) -> np.ndarray:
    """
    Darken everything except the swimmer's block + body.

    "Ignore other lanes" in the prompt does not work when four lanes are in
    the picture — so make the other lanes visually secondary instead. A soft
    (blurred) edge avoids a hard artificial border the LM might comment on.
    """
    if factor >= 0.999 or (block_cnt is None and swimmer_rect is None):
        return crop
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if block_cnt is not None:
        cv2.drawContours(mask, [block_cnt.astype(np.int32)], -1, 255, thickness=-1)
        k = max(9, int(0.16 * max(h, w)) | 1)
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    if swimmer_rect is not None:
        x1, y1, x2, y2 = swimmer_rect
        pad = max(6, int(0.03 * max(h, w)))
        cv2.rectangle(
            mask,
            (max(0, x1 - pad), max(0, y1 - pad)),
            (min(w, x2 + pad), min(h, y2 + pad)),
            255,
            -1,
        )
    soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3.0, 0.02 * max(h, w)))
    keep = factor + (1.0 - factor) * (soft.astype(np.float32) / 255.0)
    return (crop.astype(np.float32) * keep[..., None]).clip(0, 255).astype(np.uint8)



def crop_rect_from_box(
    frame_shape: tuple[int, ...],
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Stable crop window around the track box (block + swimmer).

    Always at least VLM_CROP_MIN_SIDE so the LM gets a usable picture.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad = int(np.clip(0.55 * max(bw, bh), 60, 180))
    xa = max(0, x1 - int(pad * 1.35))
    ya = max(0, y1 - pad)
    xb = min(w, x2 + pad)
    yb = min(h, y2 + int(pad * 0.85))

    # Enforce a usable minimum size (centered on the box, left-biased).
    min_side = min(VLM_CROP_MIN_SIDE, w, h)
    cw = xb - xa
    ch = yb - ya
    if cw < min_side or ch < min_side:
        cx = (x1 + x2) // 2 - int(0.08 * bw)
        cy = (y1 + y2) // 2
        need_w = max(cw, min_side)
        need_h = max(ch, min_side)
        xa = int(cx - need_w / 2)
        ya = int(cy - need_h / 2)
        xb = xa + need_w
        yb = ya + need_h
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

    # Cap oversized crops.
    max_w = min(w, max(VLM_CROP_MAX_SIDE, int(VLM_CROP_MAX_SIDE * 1.15)))
    max_h = min(h, max(VLM_CROP_MAX_SIDE, int(VLM_CROP_MAX_SIDE * 1.15)))
    if (xb - xa) > max_w or (yb - ya) > max_h:
        cx = (xa + xb) // 2 - int(0.05 * (xb - xa))
        cy = (ya + yb) // 2
        half_w = min(max_w, xb - xa) // 2
        half_h = min(max_h, yb - ya) // 2
        half_w = max(half_w, min_side // 2)
        half_h = max(half_h, min_side // 2)
        xa = max(0, min(cx - half_w, w - 2 * half_w))
        ya = max(0, min(cy - half_h, h - 2 * half_h))
        xb = min(w, xa + 2 * half_w)
        yb = min(h, ya + 2 * half_h)

    if xb <= xa or yb <= ya:
        return 0, 0, w, h
    return int(xa), int(ya), int(xb), int(yb)



# --- NEPOUZIVANE (zadne volani v projektu) — ponechano zakomentovane pro pripadne oziveni ---
# def annotate_starting_block(
#     crop_bgr: np.ndarray,
#     anchor_xy: tuple[float, float] | None = None,
# ) -> np.ndarray:
#     """
#     Outline ONE primary starting BLOCK (yellow) for the vision LM.
#
#     Prefer the block nearest the tracked swimmer's feet (anchor_xy).
#     """
#     out = crop_bgr.copy()
#     h, w = out.shape[:2]
#     red = red_block_mask(out)
#     cnt = primary_block_contour(red, anchor_xy=anchor_xy)
#     if cnt is not None:
#         cv2.drawContours(out, [cnt], -1, (0, 255, 255), 2)
#         x, y, bw, bh = cv2.boundingRect(cnt)
#         cv2.putText(
#             out,
#             "BLOCK",
#             (x, max(16, y - 6)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             (0, 255, 255),
#             2,
#             cv2.LINE_AA,
#         )
#     # Skip legend on tiny crops — it was covering the whole image.
#     if h >= 220 and w >= 220:
#         cv2.putText(
#             out,
#             "ON = foot on yellow BLOCK top",
#             (8, h - 28),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.45,
#             (0, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )
#         cv2.putText(
#             out,
#             "LEFT = yellow BLOCK empty",
#             (8, h - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.45,
#             (0, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )
#     return out
#
#
# _VLM_REF_CACHE: dict[str, str] | None = None
#
#
# --- konec nepouzivaneho bloku ---

# --- NEPOUZIVANE (zadne volani v projektu) — ponechano zakomentovane pro pripadne oziveni ---
# def read_padded_crop(
#     frame: np.ndarray,
#     box: tuple[int, int, int, int],
#     pad: int | None = None,
# ) -> np.ndarray:
#     """Backward-compatible helper: one stable crop from the track box."""
#     del pad
#     return extract_crop(frame, crop_rect_from_box(frame.shape, box))
#
#
#
# --- konec nepouzivaneho bloku ---

