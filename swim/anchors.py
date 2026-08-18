"""Pool-plane anchors from lane-rope markers -> per-frame homography.

World Aquatics ropes carry red markers at KNOWN distances from the wall
(user-confirmed and measured on test2): a solid red zone for the last 5 m
(measured 5.1 m), a small patch at 15 m (~0.5-0.8 m long) and a larger
patch at 25 m (~1.2-2.1 m). Every marker lies IN the water plane, so a
frame with the rope family (2.5 m spacing) plus at least two distinct
marker columns pins the full image->pool homography — including the
along-pool foreshortening that a scalar px/m provably cannot express.

Between such keyframes the anchor columns are PROPAGATED by the validated
camera track (translation is fine for moving a line a few frames; it was
only insufficient for converting positions to meters directly).
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from .ropes import RopeLine

# Marker classes: pool X measured from the NEAR wall of that end.
MARK_5M_MIN_LEN = 3.0     # solid end zone, >= 3 m of visible red
MARK_25M_RANGE = (1.0, 2.6)
MARK_15M_RANGE = (0.35, 1.0)


class RopeMark(NamedTuple):
    """One red run on one rope."""

    rope_index: int
    x_start: int
    x_end: int
    length_m: float
    truncated: bool  # touches a frame edge -> length unusable


def red_runs_on_rope(
    frame_hsv: np.ndarray,
    ropes: list[RopeLine],
    rope_index: int,
    band: int = 3,
) -> list[RopeMark]:
    """Red runs along one fitted rope, lengths via the local rope gap."""
    rope = ropes[rope_index]
    h, w = frame_hsv.shape[:2]
    xs = np.arange(0, w, 2)
    red = np.zeros(len(xs))
    for i, x in enumerate(xs):
        y = int(round(rope.y_at(float(x))))
        if band <= y < h - band:
            s = frame_hsv[y - band : y + band + 1, x]
            sat = (s[:, 1] > 80) & (s[:, 2] > 60)
            red[i] = (((s[:, 0] <= 12) | (s[:, 0] >= 168)) & sat).mean()
    solid = np.convolve(red, np.ones(10) / 10, "same") > 0.3

    def local_ppm(x: float) -> float | None:
        gaps = sorted(
            abs(r.y_at(x) - rope.y_at(x))
            for j, r in enumerate(ropes)
            if j != rope_index
        )
        return gaps[0] / 2.5 if gaps else None

    marks: list[RopeMark] = []
    i = 0
    while i < len(xs):
        if solid[i]:
            j = i
            gap = 0
            while j < len(xs) - 1 and gap <= 8:
                j += 1
                gap = gap + 1 if not solid[j] else 0
            j -= gap
            x0, x1 = int(xs[i]), int(xs[j])
            ppm = local_ppm(0.5 * (x0 + x1))
            if ppm and ppm > 2 and x1 > x0:
                marks.append(
                    RopeMark(
                        rope_index,
                        x0,
                        x1,
                        (x1 - x0) / ppm,
                        truncated=(x0 <= 4 or x1 >= w - 6),
                    )
                )
            i = j + gap + 1
        else:
            i += 1
    return marks


def classify_mark(mark: RopeMark) -> str | None:
    """5m / 25m / 15m by measured length; truncated runs are unusable."""
    if mark.truncated:
        return None
    if mark.length_m >= MARK_5M_MIN_LEN:
        return "5m"
    lo, hi = MARK_25M_RANGE
    if lo <= mark.length_m <= hi:
        return "25m"
    lo, hi = MARK_15M_RANGE
    if lo <= mark.length_m < hi:
        return "15m"
    return None


class AnchorColumn(NamedTuple):
    """A fitted image line of same-pool-X marker points across ropes."""

    kind: str                # '5m' | '15m' | '25m'
    coef: tuple[float, float]  # x = a*y + b
    points: int
    rms: float


def anchor_columns(
    frame_bgr: np.ndarray,
    ropes: list[RopeLine],
) -> list[AnchorColumn]:
    """Classified marker columns in one frame (needs >= 2 marked ropes)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    w_img = frame_bgr.shape[1]
    by_kind: dict[str, list[tuple[float, float]]] = {}
    for k in range(len(ropes)):
        for mark in red_runs_on_rope(hsv, ropes, k):
            kind = classify_mark(mark)
            if kind is None:
                continue
            # 15m and 5m marks exist at BOTH ends of the pool — label the
            # side, otherwise a column fit mixes left- and right-end marks
            # into one nonsense line.
            if kind in ("5m", "15m"):
                side = "L" if 0.5 * (mark.x_start + mark.x_end) < w_img / 2 else "R"
                kind = kind + side
            # Anchor x: the 5m zone is anchored at its POOL-side start (the
            # 5 m boundary); patches at their center.
            if kind.startswith("5m"):
                # A full (non-truncated) 5 m zone has TWO anchors: the inner
                # boundary at X=5 m and the wall edge at X=0 m — the wall
                # line comes for free.
                side = kind[-1]
                inner = float(mark.x_end) if side == "L" else float(mark.x_start)
                wall = float(mark.x_start) if side == "L" else float(mark.x_end)
                for kk, xx in ((kind, inner), ("0m" + side, wall)):
                    yy = float(ropes[mark.rope_index].y_at(xx))
                    by_kind.setdefault(kk, []).append((xx, yy))
                continue
            x = 0.5 * (mark.x_start + mark.x_end)
            y = float(ropes[mark.rope_index].y_at(x))
            by_kind.setdefault(kind, []).append((x, y))

    columns: list[AnchorColumn] = []
    for kind, pts in by_kind.items():
        if len(pts) < 2:
            continue
        P = np.array(pts, float)
        A = np.vstack([P[:, 1], np.ones(len(P))]).T
        coef, *_ = np.linalg.lstsq(A, P[:, 0], rcond=None)
        resid = P[:, 0] - A @ coef
        keep = np.abs(resid) < max(2.5 * np.median(np.abs(resid)) + 1, 15)
        if keep.sum() < 2:
            continue
        coef, *_ = np.linalg.lstsq(A[keep], P[keep, 0], rcond=None)
        rms = float(np.sqrt(np.mean((P[keep, 0] - A[keep] @ coef) ** 2)))
        if rms > 25:
            continue
        columns.append(
            AnchorColumn(kind, (float(coef[0]), float(coef[1])), int(keep.sum()), rms)
        )
    return columns


def anchor_columns_accumulated(
    capture: cv2.VideoCapture,
    t0: float,
    fps: float,
    ropes0: list[RopeLine],
    cam_at,
    span: float = 0.5,
    step: int = 3,
) -> list[AnchorColumn]:
    """Columns fitted from marks ACCUMULATED over +-span seconds.

    Two-point columns made the homography scale ~18% off; accumulating a
    handful of frames (marks aligned into t0's coordinates via the camera
    track) multiplies the points and lengthens the fit baseline.
    """
    all_pts: dict[str, list[tuple[float, float]]] = {}

    def collect(frame_bgr, ropes, dx):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        w_img = frame_bgr.shape[1]
        for k in range(len(ropes)):
            for mark in red_runs_on_rope(hsv, ropes, k):
                kind = classify_mark(mark)
                if kind is None:
                    continue
                if kind in ("5m", "15m"):
                    side = "L" if 0.5 * (mark.x_start + mark.x_end) < w_img / 2 else "R"
                    kind = kind + side
                if kind.startswith("5m"):
                    side = kind[-1]
                    pairs = (
                        (kind, float(mark.x_end if side == "L" else mark.x_start)),
                        ("0m" + side, float(mark.x_start if side == "L" else mark.x_end)),
                    )
                else:
                    pairs = ((kind, 0.5 * (mark.x_start + mark.x_end)),)
                for kk, xx in pairs:
                    yy = float(ropes[k].y_at(xx))
                    all_pts.setdefault(kk, []).append((xx + dx, yy))

    cam0 = cam_at(t0)
    for k in range(-int(span * fps), int(span * fps) + 1, step):
        t = t0 + k / fps
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = capture.read()
        if not ok:
            continue
        d = cam_at(t) - cam0
        ropes_t = [RopeLine(r.slope, r.intercept + r.slope * d, r.score) for r in ropes0]
        collect(frame, ropes_t, d)

    columns: list[AnchorColumn] = []
    for kind, pts in all_pts.items():
        if len(pts) < 3:
            continue
        P = np.array(pts, float)
        A = np.vstack([P[:, 1], np.ones(len(P))]).T
        coef, *_ = np.linalg.lstsq(A, P[:, 0], rcond=None)
        resid = P[:, 0] - A @ coef
        keep = np.abs(resid) < max(2.5 * np.median(np.abs(resid)) + 1, 15)
        if keep.sum() < 3:
            continue
        coef, *_ = np.linalg.lstsq(A[keep], P[keep, 0], rcond=None)
        rms = float(np.sqrt(np.mean((P[keep, 0] - A[keep] @ coef) ** 2)))
        if rms > 25:
            continue
        columns.append(
            AnchorColumn(kind, (float(coef[0]), float(coef[1])), int(keep.sum()), rms)
        )
    return columns


def homography_from_anchors(
    ropes: list[RopeLine],
    columns: list[AnchorColumn],
    closest_lane: int,
    wall_side: str,
) -> np.ndarray | None:
    """Image->pool homography from rope lines x anchor columns.

    Pool frame: X = meters from the LEFT wall of the image's pool
    (wall_side 'right' means marker distances count from the right wall,
    so X_mark = 50 - d). Y = meters across, rope k at Y = 2.5*k.
    Needs >= 2 distinct columns and >= 2 ropes -> >= 4 points.
    """
    # Distances measured from the wall the mark belongs to; converted to a
    # single axis (X from the LEFT image wall) via the side suffix.
    def pool_x_of(kind: str) -> float:
        base = {"0m": 0.0, "5m": 5.0, "15m": 15.0, "25m": 25.0}[kind.rstrip("LR")]
        if kind == "25m":
            return 25.0
        side = kind[-1]
        return base if side == "L" else 50.0 - base

    if len(columns) < 2 or len(ropes) < 2:
        return None
    ropes = sorted(ropes, key=lambda r: r.y_at(960.0))
    img_pts: list[tuple[float, float]] = []
    pool_pts: list[tuple[float, float]] = []
    for col in columns:
        a, b = col.coef
        pool_x = pool_x_of(col.kind)
        for k, rope in enumerate(ropes):
            # intersection of rope (y = m*x + c) with column (x = a*y + b)
            m, c = rope.slope, rope.intercept
            denom = 1.0 - a * m
            if abs(denom) < 1e-6:
                continue
            y = (m * b + c) / denom
            x = a * y + b
            img_pts.append((x, y))
            pool_pts.append((pool_x, 2.5 * k))
    if len(img_pts) < 4:
        return None
    H, inliers = cv2.findHomography(
        np.array(img_pts, np.float32),
        np.array(pool_pts, np.float32),
        cv2.RANSAC,
        1.0,
    )
    if H is None or inliers is None or int(inliers.sum()) < 4:
        return None
    return H


def pool_xy(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply the homography to one image point -> pool meters."""
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


class Keyframe(NamedTuple):
    """One absolutely anchored frame of a shot."""

    time_seconds: float
    ropes: list[RopeLine]
    columns: list[AnchorColumn]
    cam_dx: float


def find_keyframes(
    capture: cv2.VideoCapture,
    cam_at,
    t_start: float,
    t_end: float,
    fps: float = 30.0,
    step_seconds: float = 1.0,
    lane_count: int = 8,
) -> list[Keyframe]:
    """Scan a shot for frames that can be absolutely anchored.

    A keyframe needs a full rope ladder plus >= 2 accumulated anchor
    columns. One-way chaining from a single keyframe drifts ~0.3 m/s
    (broadcast overlay features contaminate the pan median), so DENSITY
    of keyframes is what buys absolute accuracy between them.
    """
    from .ropes import detect_lane_ropes

    keyframes: list[Keyframe] = []
    for t0 in np.arange(t_start + 0.6, t_end - 0.6, step_seconds):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
        ok, frame = capture.read()
        if not ok:
            continue
        ropes = detect_lane_ropes(frame, lane_count=lane_count)
        if len(ropes) != lane_count + 1:
            continue
        cols = anchor_columns_accumulated(
            capture, float(t0), fps, list(ropes), cam_at
        )
        kinds = {c.kind for c in cols}
        if len(kinds) < 2:
            continue
        keyframes.append(
            Keyframe(float(t0), list(ropes), cols, float(cam_at(t0)))
        )
    return keyframes


def _pool_x_from_keyframe(
    kf: Keyframe,
    cam_now: float,
    x_img: float,
    y_img: float,
    closest_lane: int,
) -> float | None:
    """Pool X of an image point, keyframe propagated to the current pan."""
    d = cam_now - kf.cam_dx
    ropes_p = [
        RopeLine(r.slope, r.intercept + r.slope * d, r.score) for r in kf.ropes
    ]
    cols_p = [
        AnchorColumn(c.kind, (c.coef[0], c.coef[1] - d), c.points, c.rms)
        for c in kf.columns
    ]
    H = homography_from_anchors(ropes_p, cols_p, closest_lane, "right")
    if H is None:
        return None
    return pool_xy(H, x_img, y_img)[0]


def pool_x_blended(
    keyframes: list[Keyframe],
    cam_at,
    t: float,
    x_img: float,
    y_img: float,
    closest_lane: int = 8,
) -> float | None:
    """Pool X at time t, blended between the neighbouring keyframes.

    Forward propagation from the previous keyframe and backward from the
    next one each carry an opposite share of the chain drift; a linear
    blend distributes the residual — the two-keyframe experiment showed
    exactly this re-anchoring recovers turns that one-way chaining hides.
    """
    if not keyframes:
        return None
    cam_now = float(cam_at(t))
    before = [k for k in keyframes if k.time_seconds <= t]
    after = [k for k in keyframes if k.time_seconds > t]
    xs = []
    weights = []
    if before:
        k1 = before[-1]
        x1 = _pool_x_from_keyframe(k1, cam_now, x_img, y_img, closest_lane)
        if x1 is not None:
            xs.append(x1)
            weights.append(
                1.0
                if not after
                else (after[0].time_seconds - t)
                / max(after[0].time_seconds - k1.time_seconds, 1e-6)
            )
    if after:
        k2 = after[0]
        x2 = _pool_x_from_keyframe(k2, cam_now, x_img, y_img, closest_lane)
        if x2 is not None:
            xs.append(x2)
            weights.append(
                1.0
                if not before
                else (t - before[-1].time_seconds)
                / max(k2.time_seconds - before[-1].time_seconds, 1e-6)
            )
    if not xs:
        return None
    w = np.array(weights, float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return float(np.dot(xs, w / w.sum()))
