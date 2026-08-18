"""Lane geometry for side-camera pool views.

Convention used by this project for a side-view broadcast camera:
- lane 8 = closest to the camera (bottom of the frame)
- lane 1 = furthest from the camera (top of the frame)

Officials on the near deck sit even lower than lane 8, so lane assignment
only uses the vertical band between ``pool_y_top_frac`` and
``pool_y_bottom_frac``. Anything outside that band is treated as non-pool.
"""

from __future__ import annotations


def pool_y_bounds(
    frame_height: int,
    pool_y_top_frac: float,
    pool_y_bottom_frac: float,
) -> tuple[float, float]:
    """Return absolute pixel y-range of the swimming surface."""
    if frame_height <= 0:
        raise ValueError("frame_height must be positive")
    if not 0.0 <= pool_y_top_frac < pool_y_bottom_frac <= 1.0:
        raise ValueError("pool Y fractions must satisfy 0 <= top < bottom <= 1")
    top = pool_y_top_frac * frame_height
    bottom = pool_y_bottom_frac * frame_height
    return top, bottom


def lane_from_center_y(
    center_y: float,
    frame_height: int,
    lane_count: int = 8,
    *,
    near_is_bottom: bool = True,
    pool_y_top_frac: float = 0.10,
    pool_y_bottom_frac: float = 0.78,
) -> int | None:
    """
    Map a mask/box center Y to a lane number, or None if outside the pool.

    With ``near_is_bottom=True`` (default):
      top of pool band -> lane 1, bottom of pool band -> lane ``lane_count``.
    """
    if lane_count < 1:
        raise ValueError("lane_count must be >= 1")
    top, bottom = pool_y_bounds(frame_height, pool_y_top_frac, pool_y_bottom_frac)
    if center_y < top or center_y > bottom:
        return None
    span = max(bottom - top, 1.0)
    frac = (center_y - top) / span
    frac = min(max(frac, 0.0), 0.999999)
    if near_is_bottom:
        return int(frac * lane_count) + 1
    return lane_count - int(frac * lane_count)


def lane_band_y_range(
    lane: int,
    frame_height: int,
    lane_count: int = 8,
    *,
    near_is_bottom: bool = True,
    pool_y_top_frac: float = 0.10,
    pool_y_bottom_frac: float = 0.78,
) -> tuple[int, int]:
    """Return inclusive-ish pixel [y0, y1) for one lane band inside the pool."""
    if not 1 <= lane <= lane_count:
        raise ValueError(f"lane must be in 1..{lane_count}")
    top, bottom = pool_y_bounds(frame_height, pool_y_top_frac, pool_y_bottom_frac)
    span = bottom - top
    band = span / lane_count
    if near_is_bottom:
        index = lane - 1  # lane 1 at top
    else:
        index = lane_count - lane
    y0 = int(round(top + index * band))
    y1 = int(round(top + (index + 1) * band))
    return y0, y1


def pick_best_for_lane(
    detections: list[tuple[tuple[int, int, int, int], float]],
    lane: int,
    frame_height: int,
    lane_count: int = 8,
    *,
    near_is_bottom: bool = True,
    pool_y_top_frac: float = 0.10,
    pool_y_bottom_frac: float = 0.78,
) -> tuple[tuple[int, int, int, int], float] | None:
    """
    Keep the highest-confidence detection whose center falls in ``lane``.

    Each detection is ``((x1, y1, x2, y2), confidence)``.
    """
    best: tuple[tuple[int, int, int, int], float] | None = None
    for box, confidence in detections:
        x1, y1, x2, y2 = box
        center_y = 0.5 * (y1 + y2)
        assigned = lane_from_center_y(
            center_y,
            frame_height,
            lane_count,
            near_is_bottom=near_is_bottom,
            pool_y_top_frac=pool_y_top_frac,
            pool_y_bottom_frac=pool_y_bottom_frac,
        )
        if assigned != lane:
            continue
        if best is None or confidence > best[1]:
            best = (box, confidence)
    return best
