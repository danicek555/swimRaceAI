"""swimRaceAI — overlay."""

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
from .hands import *  # noqa: F401,F403


def burn_reaction_overlay(
    raw_path: Path,
    out_path: Path,
    fps: float,
    beep_time: float | None,
    move_time: float | None,
    reaction: float | None,
    hand_entry: float | HandEntryResult | None = None,
    time_offset: float = 0.0,
) -> None:
    """Add beep / leave / RT / hand-entry text onto the tracked video, then H.264 encode.

    time_offset: absolute clip time of the video's first frame — the raw
    track starts at the box frame, not at 0, and the burned-in times must
    stay absolute so they match beep/leave values.
    """
    capture = cv2.VideoCapture(str(raw_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with tempfile.TemporaryDirectory() as tmp:
        marked_path = Path(tmp) / "marked.mp4"
        writer = cv2.VideoWriter(
            str(marked_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = time_offset + index / fps
            lines = [f"t={t:.2f}s"]
            if beep_time is not None:
                lines.append(f"beep {beep_time:.2f}s")
            if move_time is not None:
                lines.append(f"leave {move_time:.2f}s")
            if reaction is not None:
                lines.append(f"RT {reaction:.2f}s")
            hand_t = None
            hand_lo = hand_hi = None
            if isinstance(hand_entry, HandEntryResult):
                hand_t = hand_entry.time
                hand_lo, hand_hi = hand_entry.t_lo, hand_entry.t_hi
            elif isinstance(hand_entry, (int, float)):
                hand_t = float(hand_entry)
            if hand_t is not None:
                if (
                    hand_lo is not None
                    and hand_hi is not None
                    and abs(hand_hi - hand_lo) > 0.02
                ):
                    lines.append(f"hands {hand_lo:.2f}-{hand_hi:.2f}s")
                else:
                    lines.append(f"hands {hand_t:.2f}s")
                if move_time is not None:
                    lines.append(f"flight {hand_t - move_time:.2f}s")
            y = 28
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                y += 28
            # Red flash on the beep, green on leave-block, cyan on hand entry.
            # Beep flash spans 2 frames so it is visible at playback speed.
            if beep_time is not None and abs(t - beep_time) < (1.5 / fps):
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
                cv2.putText(
                    frame,
                    "BEEP",
                    (width // 2 - 60, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    3,
                )
            if move_time is not None and abs(t - move_time) < (0.5 / fps):
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 0), 8)
            flash_t = None
            if isinstance(hand_entry, HandEntryResult):
                flash_t = hand_entry.time
            elif isinstance(hand_entry, (int, float)):
                flash_t = float(hand_entry)
            if flash_t is not None and abs(t - flash_t) < (0.5 / fps):
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (255, 255, 0), 8)
            writer.write(frame)
            index += 1
        capture.release()
        writer.release()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(marked_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )



