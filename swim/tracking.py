"""swimRaceAI — tracking."""

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
from .blocks import *  # noqa: F401,F403


def click_swimmer(
    video_path: Path,
    start_index: int = 0,
) -> tuple[list[int] | None, int]:
    """
    Open a window, pick a frame, then draw a TIGHT box around ONE swimmer.

    Returns (box, frame_index). SAM is seeded ON THE FRAME THE BOX WAS DRAWN
    — a box drawn on frame N used to be applied to frame 0, where the same
    spot holds a different swimmer (before the start people still move
    around), so the tracker followed the wrong lane.

    start_index: frame to open on (caller passes ~1s before the beep, when
    everyone is already set on their block).

    Keys:
      a / d  = previous / next frame
      Enter  = draw the box (click-drag), then Enter again in that tool
      Esc    = cancel
    """
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_index = min(max(start_index, 0), total - 1)

    def read_frame(index: int) -> np.ndarray | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return frame if ok else None

    frame = read_frame(frame_index)
    if frame is None:
        frame_index = 0
        frame = read_frame(0)
    if frame is None:
        capture.release()
        raise SystemExit(f"Could not read video: {video_path}")

    height, width = frame.shape[:2]
    scale = min(1280 / width, 800 / height, 1.0)
    disp_w, disp_h = int(width * scale), int(height * scale)

    window = "A/D = frame, Enter = draw a tight box on ONE swimmer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(
        "A/D to find a clear frame. Enter, then drag a TIGHT box around ONE person.\n"
        "Tracking starts FROM THE FRAME you draw on — pick one where your "
        "swimmer is already set on the block."
    )

    box = None
    while True:
        vis = cv2.resize(frame, (disp_w, disp_h))
        cv2.putText(
            vis,
            f"frame {frame_index + 1}/{total}  Enter=draw box  A/D=frame  Esc=quit",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter or Space -> draw ROI
            roi = cv2.selectROI(window, vis, showCrosshair=True, fromCenter=False)
            rx, ry, rw, rh = (int(v) for v in roi)
            if rw >= 8 and rh >= 8:
                x1 = int(rx / scale)
                y1 = int(ry / scale)
                x2 = int((rx + rw) / scale)
                y2 = int((ry + rh) / scale)
                box = [x1, y1, x2, y2]
                break
            print("Box was too small. Draw around the whole person, but only that person.")
        if key == 27:
            box = None
            break
        if key in (ord("d"), ord("D")):
            frame_index = min(frame_index + 5, total - 1)
            nxt = read_frame(frame_index)
            if nxt is not None:
                frame = nxt
        if key in (ord("a"), ord("A")):
            frame_index = max(frame_index - 5, 0)
            nxt = read_frame(frame_index)
            if nxt is not None:
                frame = nxt

    capture.release()
    cv2.destroyAllWindows()
    return box, frame_index



def keep_one_person_mask(
    mask: np.ndarray,
    predicted_xy: tuple[float, float],
    max_area: float | None,
    max_jump_px: float,
) -> np.ndarray:
    """
    Keep only the blob closest to where THIS swimmer should be now.

    Important: do NOT use the original click box after they leave the blocks.
    That old spot often contains a different swimmer, which causes ID switches.
    """
    binary = (mask > 0.5).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)

    px, py = predicted_xy
    h, w = binary.shape[:2]
    best_label = 0
    best_dist = float("inf")
    for label in range(1, num):
        # stats: [x, y, width, height, area]
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        # Blob center from bounding box of the component.
        bx = stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] / 2.0
        by = stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] / 2.0
        dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_label = label

    if best_label == 0:
        return np.zeros_like(mask)

    # Too far from the predicted path => SAM latched onto a neighbor. Reject.
    if best_dist > max_jump_px:
        return np.zeros_like(mask)

    cleaned = (labels == best_label).astype(np.float32)
    area = float(np.count_nonzero(cleaned))
    if max_area is not None and area > max_area * 3.5:
        return np.zeros_like(mask)
    return cleaned



def nearest_track_box(
    track_boxes: list[tuple[int, int, int, int] | None],
    frame_index: int,
) -> tuple[int, int, int, int] | None:
    """Use this frame's box, or the closest earlier/later box if missing."""
    if 0 <= frame_index < len(track_boxes) and track_boxes[frame_index] is not None:
        return track_boxes[frame_index]
    for delta in range(1, max(len(track_boxes), 1)):
        for idx in (frame_index - delta, frame_index + delta):
            if 0 <= idx < len(track_boxes) and track_boxes[idx] is not None:
                return track_boxes[idx]
    return None



def nearest_frame_sample(
    samples: list[tuple[float, float, float] | None],
    frame_index: int,
) -> tuple[float, float, float] | None:
    """This frame's (cx, cy, foot_y) sample, or the closest frame that has one."""
    if not samples:
        return None
    if 0 <= frame_index < len(samples) and samples[frame_index] is not None:
        return samples[frame_index]
    for delta in range(1, len(samples)):
        for idx in (frame_index - delta, frame_index + delta):
            if 0 <= idx < len(samples) and samples[idx] is not None:
                return samples[idx]
    return None



def dive_direction_sign(
    samples: list[tuple[float, float, float] | None],
    fps: float,
    beep_time: float,
) -> int:
    """+1 when the swimmer dives towards larger x (right), -1 towards smaller x.

    Compares the median body-center x before the beep with the median well
    after it. Falls back to +1 (blocks left, water right — this footage).
    """
    if fps <= 0 or not samples:
        return 1
    pre = [s[0] for i, s in enumerate(samples) if s is not None and i / fps < beep_time]
    post = [s[0] for i, s in enumerate(samples) if s is not None and i / fps > beep_time + 0.6]
    if len(pre) < 2 or len(post) < 2:
        return 1
    delta = float(np.median(np.asarray(post))) - float(np.median(np.asarray(pre)))
    if abs(delta) < 8.0:
        return 1
    return 1 if delta > 0 else -1



def dive_velocity_px(
    samples: list[tuple[float, float, float] | None],
    fps: float,
    leave_time: float,
    window: float = 0.30,
) -> tuple[float, float]:
    """Estimate (vx, vy) px/s from good samples just after leave."""
    if fps <= 0 or not samples:
        return 0.0, 0.0
    pts: list[tuple[float, float, float]] = []
    for i, s in enumerate(samples):
        if s is None:
            continue
        t = i / fps
        if leave_time - 0.02 <= t <= leave_time + window:
            pts.append((t, float(s[0]), float(s[1])))
    if len(pts) < 2:
        return 0.0, 0.0
    t0, x0, y0 = pts[0]
    t1, x1, y1 = pts[-1]
    dt = max(t1 - t0, 1e-3)
    return (x1 - x0) / dt, (y1 - y0) / dt



