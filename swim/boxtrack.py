"""Temporal filter over raw SAM boxes -> a stable crop for stroke analysis.

The raw SAM 2 box is right-lane / right-swimmer most of the time, but its
GEOMETRY is noisy: it bloats with the wake behind the swimmer, collapses to
thin slivers, and drops out for a few frames. Pose estimation downstream
needs the opposite: a smooth center, a slowly changing scale and no holes.

Physics used here:
- a swimmer moves smoothly (~2 m/s) -> EMA + short prediction bridges holes,
- a swimmer is at most ~2.2 m long -> anything longer is wake; the lane
  geometry (rope lines are ~2.5 m apart) converts pixels to meters locally,
  so the cap works at any depth of a perspective shot,
- the wake is always BEHIND the swimmer -> trim the trailing side only.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import cv2
import numpy as np

from .config import (
    BOX_CENTER_ALPHA,
    BOX_MAX_HEIGHT_LANES,
    BOX_MAX_LEN_M,
    BOX_PREDICT_MAX_SECONDS,
    BOX_SIZE_ALPHA,
    BOX_SIZE_GATE,
    LANE_WIDTH_M,
    NO_POSE_CONTRAST_RATIO,
    NO_POSE_FOAM_FRAC,
)
from .ropes import RopeLine


class SmoothBox(NamedTuple):
    """One filtered box sample."""

    box: tuple[int, int, int, int] | None
    state: str  # TRACKING | PREDICTED | LOST
    length_m: float | None


def lane_px_per_meter(
    ropes: list[RopeLine] | None,
    lane: int,
    closest_lane: int,
    x: float,
) -> float | None:
    """Local image scale from the lane geometry: rope gap = LANE_WIDTH_M."""
    if ropes is None:
        return None
    lane_count = len(ropes) - 1
    if lane_count < 1 or not 1 <= lane <= lane_count:
        return None
    band = lane - 1 if closest_lane == lane_count else lane_count - lane
    top = ropes[band].y_at(x)
    bottom = ropes[band + 1].y_at(x)
    gap = abs(bottom - top)
    if gap < 4:
        return None
    return gap / LANE_WIDTH_M


class BoxTrack:
    """EMA + gating + physical caps over raw per-frame boxes.

    time_direction is -1 when the clip is time-reversed (the backward SAM
    pass): the wake then appears on the other side of the moving swimmer,
    and the trailing-side trim must follow REAL time, not clip time.
    """

    def __init__(self, fps: float, time_direction: int = 1) -> None:
        self.fps = max(fps, 1.0)
        self.time_direction = 1 if time_direction >= 0 else -1
        self.cx: float | None = None
        self.cy: float | None = None
        self.w: float | None = None
        self.h: float | None = None
        self.vx = 0.0
        self.vy = 0.0
        self.missed = 0
        self.recent_w: deque[float] = deque(maxlen=int(self.fps))
        self.recent_h: deque[float] = deque(maxlen=int(self.fps))

    def _capped_raw(
        self,
        raw: tuple[int, int, int, int],
        ropes: list[RopeLine] | None,
        lane: int,
        closest_lane: int,
    ) -> tuple[float, float, float, float, float | None]:
        """Apply the meter caps to the raw box. Returns cx, cy, w, h, len_m."""
        x1, y1, x2, y2 = (float(v) for v in raw)
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        px_per_m = lane_px_per_meter(ropes, lane, closest_lane, cx)
        length_m: float | None = None
        if px_per_m is not None:
            length_m = w / px_per_m
            max_w = BOX_MAX_LEN_M * px_per_m
            if w > max_w:
                # Wake is behind the swimmer in REAL time. Keep the leading
                # edge, cut the trailing side. Without a confident direction
                # trim symmetrically.
                real_vx = self.vx * self.time_direction
                if real_vx > 0.5:
                    x1 = x2 - max_w
                elif real_vx < -0.5:
                    x2 = x1 + max_w
                else:
                    mid = 0.5 * (x1 + x2)
                    x1 = mid - 0.5 * max_w
                    x2 = mid + 0.5 * max_w
                w = max_w
                cx = 0.5 * (x1 + x2)
                length_m = BOX_MAX_LEN_M
            max_h = BOX_MAX_HEIGHT_LANES * px_per_m * LANE_WIDTH_M
            if h > max_h:
                h = max_h
        return cx, cy, w, h, length_m

    def update(
        self,
        raw: tuple[int, int, int, int] | None,
        ropes: list[RopeLine] | None,
        lane: int,
        closest_lane: int,
        frame_size: tuple[int, int],
    ) -> SmoothBox:
        """Feed one frame's raw box (or None). Returns the filtered box."""
        width, height = frame_size

        if raw is not None:
            cx, cy, w, h, length_m = self._capped_raw(
                raw, ropes, lane, closest_lane
            )
            # Gate sudden size jumps against the recent median (bloat that
            # survived the meter cap, or a sliver collapse).
            if len(self.recent_w) >= max(4, int(0.3 * self.fps)):
                med_w = float(np.median(self.recent_w))
                med_h = float(np.median(self.recent_h))
                w = float(np.clip(w, med_w / BOX_SIZE_GATE, med_w * BOX_SIZE_GATE))
                h = float(np.clip(h, med_h / BOX_SIZE_GATE, med_h * BOX_SIZE_GATE))
            self.recent_w.append(w)
            self.recent_h.append(h)

            if self.cx is None:
                self.cx, self.cy, self.w, self.h = cx, cy, w, h
            else:
                dx = cx - self.cx
                dy = cy - self.cy
                self.vx = 0.8 * self.vx + 0.2 * dx
                self.vy = 0.8 * self.vy + 0.2 * dy
                self.cx += BOX_CENTER_ALPHA * dx
                self.cy += BOX_CENTER_ALPHA * dy
                self.w += BOX_SIZE_ALPHA * (w - self.w)
                self.h += BOX_SIZE_ALPHA * (h - self.h)
            self.missed = 0
            state = "TRACKING"
        else:
            if self.cx is None:
                return SmoothBox(None, "LOST", None)
            self.missed += 1
            if self.missed > BOX_PREDICT_MAX_SECONDS * self.fps:
                return SmoothBox(None, "LOST", None)
            # Coast on the smoothed velocity; keep the size frozen.
            self.cx += self.vx
            self.cy += self.vy
            state = "PREDICTED"
            length_m = None

        x1 = int(round(max(0.0, self.cx - 0.5 * self.w)))
        y1 = int(round(max(0.0, self.cy - 0.5 * self.h)))
        x2 = int(round(min(float(width - 1), self.cx + 0.5 * self.w)))
        y2 = int(round(min(float(height - 1), self.cy + 0.5 * self.h)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return SmoothBox(None, "LOST", None)
        if state == "TRACKING" and length_m is None and self.w is not None:
            px_per_m = lane_px_per_meter(ropes, lane, closest_lane, self.cx)
            if px_per_m is not None:
                length_m = self.w / px_per_m
        return SmoothBox((x1, y1, x2, y2), state, length_m)


def no_pose_flag(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    lane_mask: np.ndarray | None,
) -> bool:
    """True when the crop is a bad candidate for pose (underwater / glide).

    Underwater phases show almost no foam and little contrast against the
    surrounding lane water — pose estimation there wastes effort and returns
    noise, so the segment is only MARKED, not fought.
    """
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return True
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    foam = float(np.mean((hsv[:, :, 2] >= 190) & (hsv[:, :, 1] <= 95)))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    inside = float(np.std(gray[y1:y2, x1:x2]))
    if lane_mask is not None:
        outside_mask = lane_mask.astype(bool).copy()
        outside_mask[y1:y2, x1:x2] = False
        outside_vals = gray[outside_mask]
        outside = float(np.std(outside_vals)) if outside_vals.size > 400 else 1.0
    else:
        outside = 1.0
    contrast_ratio = inside / max(outside, 1e-3)
    return foam < NO_POSE_FOAM_FRAC and contrast_ratio < NO_POSE_CONTRAST_RATIO
