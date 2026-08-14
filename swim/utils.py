"""swimRaceAI — utils."""

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


def log_time(label: str, start: float) -> None:
    """Print how many seconds a step took, like: YOLO detect  41.2s"""
    print(f"  {label}: {time.perf_counter() - start:.1f}s")



def format_hms(seconds: float) -> str:
    """Turn 95.2 into '1m 35s' so ETAs are easy to read."""
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    if minutes == 0:
        return f"{secs}s"
    return f"{minutes}m {secs}s"



def list_videos(folder: Path) -> list[Path]:
    """Return every video file in the folder, sorted by name."""
    videos = []
    for path in folder.iterdir():
        # Skip folders and files that are not videos.
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return sorted(videos)



def pick_video(folder: Path, video_name: str | None) -> list[Path]:
    """
    If the user passed -v, return that one video.
    If they did not, return every video in the folder.
    """
    videos = list_videos(folder)
    if not videos:
        raise SystemExit(f"No video files found in: {folder}")

    # No -v flag: process the whole folder.
    if video_name is None:
        return videos

    # -v can be a filename like race1.mp4, or a full path.
    requested = Path(video_name)
    if requested.exists() and requested.is_file():
        return [requested]

    matches = [video for video in videos if video.name == requested.name]
    if not matches:
        names = ", ".join(video.name for video in videos)
        raise SystemExit(
            f"Could not find '{video_name}' in {folder}. Videos there: {names}"
        )
    return matches



def clip_frame_count(clip_path: Path) -> tuple[int, float]:
    """How many frames and what fps the clip has."""
    capture = cv2.VideoCapture(str(clip_path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
    capture.release()
    if frames <= 0:
        frames = int(TRACK_SECONDS * fps)
    return frames, fps



def extract_crop(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    xa, ya, xb, yb = rect
    return frame[ya:yb, xa:xb].copy()



def clip_video_start(input_path: Path, clip_path: Path) -> None:
    """Copy the first DETECT_SECONDS, cropped to the left (starting blocks)."""
    command = [
        "ffmpeg",
        "-y",  # overwrite the output file if it already exists
        "-i", str(input_path),
        "-t", str(DETECT_SECONDS),  # stop after 10 seconds
        "-an",  # no audio; YOLO only needs the picture
        # crop=width:height:x:y
        # iw*0.32 = left 32% of the frame, full height, starting at x=0.
        "-vf", f"crop=iw*{BLOCK_CROP_WIDTH}:ih:0:0",
        str(clip_path),
    ]
    subprocess.run(command, check=True, capture_output=True)



def clip_first_seconds(input_path: Path, clip_path: Path, seconds: int) -> None:
    """Copy only the first N seconds. Full frame, no crop (needed for click-to-track)."""
    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-t", str(seconds),
        "-an",
        str(clip_path),
    ]
    subprocess.run(command, check=True, capture_output=True)



