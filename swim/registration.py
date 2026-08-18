"""Pool-coordinate registration: subtract camera pan from image positions.

Image motion = swimmer motion + camera motion. The rope ladder already pins
vertical pan/zoom/roll (ropes are pool-fixed lines), but sliding ALONG the
ropes is invisible to them — that one missing degree of freedom is what
biased speeds and hid turns on panning shots.

Mechanism (probe-selected): median optical flow of features OUTSIDE the
water (deck, stands, structures — ``pool_water_mask`` inverted). Measured
on test2: static start 0.00 px, panning shots MAD < 2 px, works at the
100 m turn where rope fitting fails entirely; a synthetic ±40 px pan is
recovered with 0.00 px error. The bead-phase alternative was probed first
and rejected: broadcast 1080p blurs the beads (no correlation lock).

The cumulative pan restarts at every camera cut — downstream analysis
already processes blocks per shot, so coordinates only need to be
pool-fixed WITHIN a shot; absolute anchoring comes from wall touches.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .ropes import pool_water_mask

REG_MAX_CORNERS = 400
REG_MIN_POINTS = 25
REG_MAX_MAD_PX = 4.0


def _pair_motion(
    gray_prev: np.ndarray,
    gray_now: np.ndarray,
    static_mask: np.ndarray,
) -> tuple[float, int, float] | None:
    """Median scene flow between two frames on static (non-water) regions.

    Returns (scene_dx, points, mad) or None when unreliable.
    """
    pts = cv2.goodFeaturesToTrack(
        gray_prev,
        maxCorners=REG_MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=12,
        mask=static_mask,
    )
    if pts is None or len(pts) < REG_MIN_POINTS:
        return None
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_now, pts, None, winSize=(21, 21), maxLevel=3
    )
    good = st.ravel() == 1
    if good.sum() < REG_MIN_POINTS:
        return None
    d = (nxt[good] - pts[good]).reshape(-1, 2)
    med = float(np.median(d[:, 0]))
    mad = float(np.median(np.abs(d[:, 0] - med)))
    if mad > REG_MAX_MAD_PX:
        return None
    return med, int(good.sum()), mad


def camera_track(
    video_path: Path,
    t_start: float,
    t_end: float,
    cut_times: list[float] | None = None,
    fps: float | None = None,
) -> dict[str, np.ndarray]:
    """Cumulative camera pan per frame over [t_start, t_end).

    cam_dx[i] is the camera's accumulated horizontal pan in px since the
    START OF ITS SHOT; pool-fixed x = image_x + cam_dx. The chain resets at
    every cut in ``cut_times`` (a cut is a new camera = a new frame of
    reference). Frames where the flow is unreliable coast on the previous
    per-frame pan (marked low confidence).
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    if fps is None:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    cuts = sorted(cut_times or [])

    first = int(round(t_start * fps))
    last = int(round(t_end * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    times: list[float] = []
    cam_dx: list[float] = []
    confident: list[int] = []

    prev_gray: np.ndarray | None = None
    prev_mask: np.ndarray | None = None
    cum = 0.0
    last_step = 0.0
    next_cut_i = 0
    while cuts and next_cut_i < len(cuts) and cuts[next_cut_i] <= t_start + 1e-6:
        next_cut_i += 1

    for fi in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        t = fi / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        water = pool_water_mask(frame)
        mask = cv2.erode(
            (water == 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8)
        )

        if next_cut_i < len(cuts) and t >= cuts[next_cut_i] - 1e-6:
            # New shot = new camera frame of reference.
            cum = 0.0
            last_step = 0.0
            prev_gray = None
            next_cut_i += 1

        conf = 1
        if prev_gray is not None:
            pair = _pair_motion(prev_gray, gray, prev_mask)
            if pair is not None:
                scene_dx, _n, _mad = pair
                last_step = -scene_dx  # camera pan is minus the scene flow
            else:
                conf = 0  # coast on the previous pan speed
            cum += last_step
        times.append(t)
        cam_dx.append(cum)
        confident.append(conf)
        prev_gray = gray
        prev_mask = mask

    capture.release()
    return {
        "t": np.asarray(times),
        "cam_dx": np.asarray(cam_dx),
        "confident": np.asarray(confident),
    }


def write_camera_track_csv(track: dict[str, np.ndarray], path: Path) -> None:
    lines = ["time_s,cam_dx_px,confident"]
    for t, dx, c in zip(track["t"], track["cam_dx"], track["confident"]):
        lines.append(f"{t:.3f},{dx:.2f},{int(c)}")
    path.write_text("\n".join(lines) + "\n")
