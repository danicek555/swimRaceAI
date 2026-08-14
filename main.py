# =============================================================================
# HOW THIS PROGRAM WORKS (big picture)
#
# You already have videos on your computer. You point this script at a folder.
# It does NOT download anything.
#
# Example:
#   python main.py -f videos
#   python main.py -f videos -v race1.mp4
#
# Step by step:
#   1. main() reads -f (folder) and optional -v (one video name).
#   2. run() checks the folder exists, then pick_video() decides which files
#      to look at: one video, or every video in the folder.
#   3. For each video, process_video() does the real work:
#        a) extract_audio() uses ffmpeg to copy ONLY the first 15 seconds of
#           SOUND into a simple .wav file. We ignore the picture.
#        b) find_beep() looks at that sound and answers: "at what second
#           did the start beep begin?"
#        c) detect_people() uses YOLO on the PICTURE, but ONLY the first
#           10 seconds. It draws a box around every person it finds, and
#           saves a new video in the output/ folder.
#   4. We print the beep time, like: The beep occurred at 1.840 seconds
#
# Why look at sound, not the video picture?
#   A start beep is a short, loud, high tone. That is much easier to find in
#   the audio than by watching pixels.
#
# How find_beep() decides "this is the beep":
#   - A WAV file is a long list of numbers (samples). At 44100 samples/second,
#     sample 44100 is exactly 1.000 second.
#   - We keep only frequencies between 800 Hz and 3000 Hz. A beep lives there.
#     A splash, rumble, or voice is more "spread out", so a lot of it gets
#     filtered away.
#   - We chop the filtered sound into 20 ms slices and measure how loud each
#     slice is (RMS loudness).
#   - The loudest slice in those 15 seconds is our guess for the beep.
#   - If that loudest slice is not at least 4x louder than typical background,
#     we say "no clear beep" instead of guessing.
#   - Then we walk BACKWARD from the loudest slice until the sound gets quiet
#     again. That is closer to when the beep STARTED, not its middle.
#   - Finally: beep_time = (slice number) * 0.020 seconds
#
# How detect_people() works (YOLO):
#   YOLO is a pretrained model: it already learned "what a person looks like"
#   from lots of photos. We do NOT train it ourselves.
#   For each video frame (one still image from the video) YOLO answers:
#     - is there a person here?
#     - where is the box around them? (x, y, width, height)
#     - how sure is it? (confidence, 0.0 to 1.0)
#   We only keep class 0, which in the COCO dataset means "person".
#   We only run this on the first 10 seconds, so a long race stays fast.
#   Extra tricks so crouched swimmers on the blocks get boxes:
#     - YOLO26 medium DETECT model (boxes, not pose — pose was missing crouches)
#     - YOLO-World open-vocab: look for "swimmer" / "person on starting block"
#     - crop the LEFT side, contrast-boost, split into small overlapping TILES
#     - drop only CLEARLY standing officials and people already in the water
#     - do NOT require a perfect crouch or red overlap (that deleted real boxes)
#   Important: this is DETECTION (a person is here), not "who is this swimmer".
# =============================================================================

# argparse reads flags from the terminal, like:
#   python main.py -f videos
#   python main.py -f videos -v race1.mp4
import argparse
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

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from ultralytics import YOLO

try:
    from ultralytics import YOLOWorld
except ImportError:
    YOLOWorld = None


def load_dotenv_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (if missing)."""
    env_path = path or (Path(__file__).resolve().parent / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file()

# Only look at the start of the video. The start beep should be here.
SEARCH_SECONDS = 15

# YOLO only looks at the first 10 seconds of picture. Faster on a laptop.
DETECT_SECONDS = 6

# Click-to-track also only uses the start of the race.
TRACK_SECONDS = 6

# Official World Aquatics "reaction time" = beep -> feet LEAVE the block
# (pressure switch), usually ~0.55-0.80s for freestyle. We approximate that
# with a BIG body move, not a 12px SAM wobble.
#
# Threshold is a FRACTION of the swimmer's on-screen size (not raw pixels),
# so zoom / distance / resolution do not need re-calibration every time.
# Optional: once tune with --known-rt 0.62, then set LEAVE_BLOCK_BODY_FRAC.
LEAVE_BLOCK_BODY_FRAC = 0.75
# Require that big move to last this long (seconds) so noise does not count.
LEAVE_HOLD_SECONDS = 0.05
# After the beep, resample motion every 0.01s (like the timing console).
RT_STEP_SECONDS = 0.01
# If the mask center jumps farther than this in one frame, treat it as an
# identity switch (neighbor steal) and keep the previous mask instead.
# At 30fps a real dive move is usually under this; a lane-jump is bigger.
MAX_CENTER_JUMP_PX = 90

# Hybrid RT: vision LM finds when feet leave the block. Needs OPENAI_API_KEY.
# Pick a vision Chat Completions model via SWIM_VLM_MODEL in .env, e.g.:
#   gpt-4o-mini | gpt-4.1-mini | gpt-4o | gpt-4.1 | gpt-5.2 | gpt-5.4 | gpt-5.6-terra
# See .env.example for the full list. Default: gpt-4o
VLM_MODEL = os.environ.get("SWIM_VLM_MODEL", "gpt-4o")
VLM_API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# Ask the LM in this full window after the beep.
VLM_RT_MIN_SECONDS = 0.50
VLM_RT_SEARCH_SECONDS = 1.20
# Dense band (typical RT): sample every VLM_SAMPLE_DENSE_STEP.
VLM_DENSE_MIN_SECONDS = 0.58
VLM_DENSE_MAX_SECONDS = 0.80
VLM_SAMPLE_DENSE_STEP_SECONDS = 0.03
# Outside the dense band (still inside 0.50–1.20): coarser samples.
VLM_SAMPLE_COARSE_STEP_SECONDS = 0.05
# Need this many LEFT_BLOCK answers in a row (no ON in between) to trust leave.
# Stops "LEFT then 0.2s later still ON" false alarms when the crop missed a foot.
VLM_LEFT_CONFIRM = 3
# Tight crop around feet + block (not the whole pool / other lanes).
# Final crop side length is clamped to this range (pixels).
VLM_CROP_MIN_SIDE = 360
VLM_CROP_MAX_SIDE = 520
# Max side length of JPEG sent to the LM.
VLM_JPEG_MAX_SIDE = 512
# Motion refine (reported on its own line; primary RT stays LM).
VLM_REFINE_SIGNAL_FRAC = 0.55
# Few-shot reference photos shown to the LM before each crop.
VLM_REFS_DIR = Path(__file__).resolve().parent / "vlm_refs"
VLM_REF_ON = VLM_REFS_DIR / "example_ON_BLOCK.jpg"
VLM_REF_ON_HARD = VLM_REFS_DIR / "example_ON_BLOCK_hard.jpg"
VLM_REF_LEFT = VLM_REFS_DIR / "example_LEFT_BLOCK.jpg"
VLM_REF_LEFT_HARD = VLM_REFS_DIR / "example_LEFT_BLOCK_hard.jpg"
VLM_REF_ON_WEDGE = VLM_REFS_DIR / "example_ON_BLOCK_wedge.jpg"
# Local foot∩block overlap: force ON_BLOCK if this many pixels touch.
VLM_OVERLAP_FORCE_ON_PX = 80

# A swim start beep is usually a mid/high tone, not a splash or a voice.
# Hz = Hertz = "how many vibrations per second". Higher Hz = higher pitch.
BEEP_MIN_HZ = 800
BEEP_MAX_HZ = 3000

# Measure loudness in tiny slices of audio (20 milliseconds each).
# Smaller windows = more precise time, but noisier measurements.
WINDOW_MS = 20

# File types we treat as videos inside the folder.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Detect model (COCO "person"), not pose. Pose missed the crouched starts.
YOLO_DETECT_MODEL = "yolo26m.pt"

# Open-vocabulary model: we tell it the words "swimmer" / "starting block".
YOLO_WORLD_MODEL = "yolov8m-worldv2.pt"
WORLD_CLASSES = ["swimmer", "person on starting block", "athlete"]

# In the COCO dataset, class 0 = person.
PERSON_CLASS_ID = 0

# Low-ish, but 0.12 was showing junk like 0.13 fragments.
PERSON_CONFIDENCE = 0.25
WORLD_CONFIDENCE = 0.20

YOLO_IMAGE_SIZE = 960
WORLD_IMAGE_SIZE = 1280

# Lower iou = merge overlapping boxes more aggressively (tiles + two models).
YOLO_IOU = 0.32

# Wider left crop so leaning bodies are not cut off.
BLOCK_CROP_WIDTH = 0.42

# Smaller tiles = more zoom on 1-2 lanes.
TILE_HEIGHT_RATIO = 0.30
TILE_OVERLAP_RATIO = 0.16

# Skip frames to keep a laptop usable (2 = every other frame, reuse last boxes).
DETECT_EVERY_N_FRAMES = 2

# Ignore people too far left (crowd/officials) or too far right (in the water).
ON_BLOCK_X_MIN = 0.12
ON_BLOCK_X_MAX = 0.60

LANE_COUNT = 8

# COCO pose is unused for the main detector now. Standing filter is optional.
MAX_UPRIGHT = 0.28
KEYPOINT_MIN_CONF = 0.25

# Soft red hint only (not a hard requirement).
RED_OVERLAP = 0.008

# Tiny SAM 2 model. First run downloads it. Follows the person you click.
SAM_TRACK_MODEL = "sam2_t.pt"

# Where we save the video with boxes drawn on it.
OUTPUT_FOLDER = Path("output")


def log_time(label: str, start: float) -> None:
    """Print how many seconds a step took, like: YOLO detect  41.2s"""
    print(f"  {label}: {time.perf_counter() - start:.1f}s")


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


def extract_audio(input_path: Path, wav_path: Path) -> None:
    """Turn the video into a short, simple WAV file with ffmpeg."""
    print(f"Extracting audio from {input_path.name}...")
    # ffmpeg is a separate program on your computer. We call it like a
    # terminal command. It reads the video and writes only the sound.
    command = [
        "ffmpeg",
        "-y",  # overwrite the output file if it already exists
        "-i", str(input_path),
        "-t", str(SEARCH_SECONDS),  # only the first 15 seconds
        "-ac", "1",  # 1 channel = mono (easier to analyze)
        "-ar", "44100",  # 44100 samples per second
        "-sample_fmt", "s16",  # a common WAV format scipy can read
        str(wav_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def find_beep(wav_path: Path) -> float | None:
    """
    Find when the start beep happens.

    Returns the time in seconds, or None if it does not find a clear beep.
    """
    # sample_rate = how many audio samples are in one second (we asked for 44100)
    # audio = a big list of numbers, one per sample
    sample_rate, audio = wavfile.read(wav_path)

    # If the file is stereo (2 columns), average the two speakers into one.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert from integers (like -32768 to 32767) into small floats (-1 to 1).
    # That way "loud" always means close to 1.0, no matter the file format.
    audio = audio.astype(np.float64)
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value

    # Keep only the first SEARCH_SECONDS (ffmpeg already trimmed, this is extra safety).
    max_samples = int(SEARCH_SECONDS * sample_rate)
    audio = audio[:max_samples]

    # Keep frequencies that sound like a beep, and remove rumble / splash.
    # A "band-pass filter" keeps a band of pitches and throws the rest away.
    # scipy wants the cutoffs as a fraction of the Nyquist frequency
    # (Nyquist = half the sample rate, here 22050 Hz).
    nyquist = sample_rate / 2
    low = BEEP_MIN_HZ / nyquist
    high = BEEP_MAX_HZ / nyquist
    filter_b, filter_a = butter(4, [low, high], btype="band")
    beep_band = filtfilt(filter_b, filter_a, audio)

    # Walk through the audio in small windows and measure loudness in each one.
    # Example: 44100 samples/sec * 0.020 sec = 882 samples per window.
    window_samples = int(sample_rate * WINDOW_MS / 1000)
    energies = []
    for start in range(0, len(beep_band) - window_samples, window_samples):
        window = beep_band[start : start + window_samples]
        # RMS = a common way to measure "how loud is this chunk?"
        # It is basically: square each sample, average them, then square-root.
        loudness = np.sqrt(np.mean(window ** 2))
        energies.append(loudness)

    energies = np.array(energies)
    if len(energies) == 0:
        return None

    # The beep should be much louder than typical background sound.
    # argmax = index of the biggest number. median = a typical/middle value,
    # which is less fooled by one huge spike than an average would be.
    peak_index = int(np.argmax(energies))
    peak = energies[peak_index]
    typical = np.median(energies)
    if peak < typical * 4:
        return None

    # Walk backward from the loudest moment to where the beep started.
    # The peak is often the MIDDLE of the beep. We want the START.
    # Keep stepping left while the slice is still at least 40% as loud as the peak.
    start_threshold = peak * 0.4
    start_index = peak_index
    while start_index > 0 and energies[start_index] > start_threshold:
        start_index -= 1

    # Convert "which window?" into seconds.
    # Window 0 = 0.000s, window 1 = 0.020s, window 92 = 1.840s, and so on.
    window_seconds = WINDOW_MS / 1000
    beep_time = start_index * window_seconds
    return beep_time


def load_yolo_model() -> dict:
    """Load detectors once. Reusing them for every video is much faster."""
    models: dict = {}
    start = time.perf_counter()

    print("Loading YOLO26m detect model...")
    models["detect"] = YOLO(YOLO_DETECT_MODEL)
    log_time("Load detect model", start)

    models["world"] = None
    if YOLOWorld is not None:
        world_start = time.perf_counter()
        print("Loading YOLO-World (open-vocab swimmer)...")
        try:
            world = YOLOWorld(YOLO_WORLD_MODEL)
            world.set_classes(WORLD_CLASSES)
            models["world"] = world
            log_time("Load YOLO-World", world_start)
        except Exception as error:
            print(f"  YOLO-World skipped: {error}")
    else:
        print("  YOLO-World not installed; using detect model only.")

    return models


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


def enhance_tile(tile: np.ndarray) -> np.ndarray:
    """Boost local contrast so dark swim caps / wet skin stand out more."""
    lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def tile_y_ranges(frame_height: int) -> list[tuple[int, int]]:
    """
    Walk down the tall block strip and return (y_start, y_end) for each tile.

    Example: height 1080, tile ~450px, overlap ~240px -> 4 stacked close-ups.
    """
    tile_height = max(int(frame_height * TILE_HEIGHT_RATIO), 1)
    overlap = int(frame_height * TILE_OVERLAP_RATIO)
    step = max(tile_height - overlap, 1)

    ranges = []
    y_start = 0
    while y_start < frame_height:
        y_end = min(y_start + tile_height, frame_height)
        ranges.append((y_start, y_end))
        if y_end == frame_height:
            break
        y_start += step
    return ranges


def boxes_from_tile(result, y_offset: int) -> list[tuple]:
    """Take YOLO boxes + keypoints from one tile and shift them onto the full frame."""
    detections = []
    if result.boxes is None:
        return detections

    kpts_xy = None
    kpts_conf = None
    kpts = getattr(result, "keypoints", None)
    if kpts is not None and getattr(kpts, "xy", None) is not None:
        kpts_xy = kpts.xy.cpu().numpy()
        if getattr(kpts, "conf", None) is not None:
            kpts_conf = kpts.conf.cpu().numpy()

    for i, box in enumerate(result.boxes):
        # xyxy = [left, top, right, bottom] in pixels inside THIS tile.
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        xy = None
        conf = None
        if kpts_xy is not None and i < len(kpts_xy):
            xy = kpts_xy[i].copy()
            xy[:, 1] += y_offset
        if kpts_conf is not None and i < len(kpts_conf):
            conf = kpts_conf[i]
        detections.append(([x1, y1 + y_offset, x2, y2 + y_offset], score, xy, conf))
    return detections


def nms_merge(detections: list[tuple[list[float], float]]) -> list[tuple[list[int], float]]:
    """If two tiles saw the same person, keep the stronger box."""
    if not detections:
        return []

    xywh = []
    scores = []
    for (x1, y1, x2, y2), score in detections:
        xywh.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(score)

    keep = cv2.dnn.NMSBoxes(xywh, scores, 0.05, YOLO_IOU)
    if len(keep) == 0:
        return []

    merged = []
    for i in np.array(keep).flatten():
        x1, y1, x2, y2 = detections[i][0]
        merged.append(([int(x1), int(y1), int(x2), int(y2)], detections[i][1]))
    return merged


def _joint_midpoint(kpts: np.ndarray, conf: np.ndarray | None, i: int, j: int):
    """Average two keypoints if YOLO is confident they exist."""
    if conf is not None and (conf[i] < KEYPOINT_MIN_CONF or conf[j] < KEYPOINT_MIN_CONF):
        return None
    points = kpts[[i, j]]
    if np.any(points <= 0):
        return None
    return points.mean(axis=0)


def is_clearly_standing(kpts: np.ndarray | None, conf: np.ndarray | None, box: list[float]) -> bool:
    """
    True only when the skeleton clearly looks like a standing official.

    If keypoints are missing (common on a side-view crouch), return False
    so we do NOT throw the detection away.
    """
    if kpts is None:
        return False
    shoulders = _joint_midpoint(kpts, conf, 5, 6)
    hips = _joint_midpoint(kpts, conf, 11, 12)
    if shoulders is None or hips is None:
        return False

    box_h = max(box[3] - box[1], 1.0)
    # Image y grows downward, so hips_y - shoulders_y is big for someone upright.
    vertical = (hips[1] - shoulders[1]) / box_h
    dx = abs(hips[0] - shoulders[0])
    dy = abs(hips[1] - shoulders[1])
    torso_is_sideways = dx > dy * 0.8
    return vertical >= MAX_UPRIGHT and not torso_is_sideways


def red_block_mask(frame: np.ndarray) -> np.ndarray:
    """White pixels = reddish starting-block color. HSV handles lighting better than RGB."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red wraps around hue 0, so we need two ranges.
    low_red = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([15, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(low_red, high_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Starting blocks are on the LEFT. Red lane lines in the water are on the RIGHT.
    # Zero the right side so people in the water do not count as "on a block".
    cutoff = int(mask.shape[1] * 0.55)
    mask[:, cutoff:] = 0
    return mask


def overlaps_red_block(red_mask: np.ndarray, box: list[float]) -> bool:
    """True if the box (and a little below it) sits on enough red (the block)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    height, width = red_mask.shape[:2]
    pad = int((y2 - y1) * 0.3)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2 + pad, height)
    if x2 <= x1 or y2 <= y1:
        return False
    roi = red_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    return float(np.count_nonzero(roi)) / roi.size >= RED_OVERLAP


def keep_likely_block_people(
    frame: np.ndarray,
    detections: list[tuple],
    frame_width: int,
    frame_height: int,
) -> list[tuple[list[float], float]]:
    """
    Clean up the pile of person boxes.

    Keep people on the starting-block strip. Drop water, tiny fragments,
    and tall skinny boxes that are usually standing officials.
    """
    red = red_block_mask(frame)
    min_area = frame_width * frame_height * 0.004
    kept = []
    for box, score, kpts, conf in detections:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if width * height < min_area:
            continue
        # Officials are usually tall and thin. Swimmers on blocks are compact.
        if height > width * 2.2:
            continue
        center_x = ((x1 + x2) / 2) / frame_width
        if center_x < ON_BLOCK_X_MIN or center_x > ON_BLOCK_X_MAX:
            continue
        if is_clearly_standing(kpts, conf, box):
            continue
        on_red = overlaps_red_block(red, box)
        in_block_column = 0.20 <= center_x <= 0.52
        if not on_red and not in_block_column:
            continue
        kept.append((box, score))
    return kept


def pick_one_per_lane(
    detections: list[tuple[list[int], float]],
    frame_height: int,
) -> list[tuple[list[int], float, int]]:
    """Keep the strongest box in each of 8 horizontal bands (one per lane)."""
    best: dict[int, tuple[list[int], float]] = {}
    for (x1, y1, x2, y2), score in detections:
        center_y = (y1 + y2) / 2
        lane = int(center_y / frame_height * LANE_COUNT) + 1
        lane = min(max(lane, 1), LANE_COUNT)
        if lane not in best or score > best[lane][1]:
            best[lane] = ([x1, y1, x2, y2], score)
    return [(box, score, lane) for lane, (box, score) in sorted(best.items())]


def run_detect_on_image(model, image: np.ndarray, y_offset: int, conf: float, imgsz: int, person_only: bool) -> list[tuple]:
    """Run one model on one image and map boxes back to the full frame."""
    kwargs = {
        "source": image,
        "conf": conf,
        "imgsz": imgsz,
        "iou": YOLO_IOU,
        "max_det": 40,
        "verbose": False,
    }
    if person_only:
        kwargs["classes"] = [PERSON_CLASS_ID]
    results = model.predict(**kwargs)
    return boxes_from_tile(results[0], y_offset)


def detect_people(video_path: Path, models: dict) -> None:
    """
    Run YOLO on the first DETECT_SECONDS, on overlapping tiles of the block strip,
    draw a box around each person, and save a new video under output/<video_name>/.
    """
    print(f"Detecting people with YOLO in the first {DETECT_SECONDS} seconds...")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Temporary folder is deleted automatically when we leave this block.
    # We clip first so YOLO never has to scan the whole race.
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / f"{video_path.stem}_first{DETECT_SECONDS}s.mp4"
        clip_start = time.perf_counter()
        clip_video_start(video_path, clip_path)
        log_time("Clip + crop video", clip_start)

        capture = cv2.VideoCapture(str(clip_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tiles = tile_y_ranges(height)

        out_dir = OUTPUT_FOLDER / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_path.stem}_first{DETECT_SECONDS}s.mp4"
        # OpenCV writes mpeg4, which Cursor/QuickTime often cannot play.
        # Write a temp file first, then ffmpeg converts it to H.264.
        raw_path = Path(tmp) / "raw_boxes.mp4"
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        max_people = 0
        frame_count = 0
        last_people: list[tuple[list[int], float]] = []
        yolo_start = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1

            should_detect = (frame_count - 1) % DETECT_EVERY_N_FRAMES == 0
            if should_detect:
                boosted = enhance_tile(frame)
                detections = []

                # Close-up tiles: regular person detector.
                for y_start, y_end in tiles:
                    tile = boosted[y_start:y_end, :]
                    detections.extend(
                        run_detect_on_image(
                            models["detect"],
                            tile,
                            y_start,
                            PERSON_CONFIDENCE,
                            YOLO_IMAGE_SIZE,
                            person_only=True,
                        )
                    )

                # Whole strip: open-vocab "swimmer" detector, if it loaded.
                if models["world"] is not None:
                    detections.extend(
                        run_detect_on_image(
                            models["world"],
                            boosted,
                            0,
                            WORLD_CONFIDENCE,
                            WORLD_IMAGE_SIZE,
                            person_only=False,
                        )
                    )

                filtered = keep_likely_block_people(boosted, detections, width, height)
                people = nms_merge(filtered)
                people = pick_one_per_lane(people, height)
                last_people = people

            people = last_people
            if len(people) > max_people:
                max_people = len(people)

            # Draw our own boxes so we control what the saved video shows.
            for (x1, y1, x2, y2), score, lane in people:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                cv2.putText(
                    frame,
                    f"L{lane} {score:.2f}",
                    (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 180, 0),
                    2,
                )
            writer.write(frame)

        log_time(f"YOLO detect ({frame_count} frames)", yolo_start)
        capture.release()
        writer.release()

        # H.264 + yuv420p is what Cursor, QuickTime, and browsers can play.
        encode_start = time.perf_counter()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(raw_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        log_time("Encode H.264 video", encode_start)

    print(f"Most people seen in one frame: {max_people}")
    print(f"Saved boxed video in: {out_path}")


def click_swimmer(video_path: Path) -> list[int] | None:
    """
    Open a window, pick a frame, then draw a TIGHT box around ONE swimmer.

    Keys:
      a / d  = previous / next frame
      Enter  = draw the box (click-drag), then Enter again in that tool
      Esc    = cancel
    """
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_index = 0

    def read_frame(index: int) -> np.ndarray | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return frame if ok else None

    frame = read_frame(0)
    if frame is None:
        capture.release()
        raise SystemExit(f"Could not read video: {video_path}")

    height, width = frame.shape[:2]
    scale = min(1280 / width, 800 / height, 1.0)
    disp_w, disp_h = int(width * scale), int(height * scale)

    window = "A/D = frame, Enter = draw a tight box on ONE swimmer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("A/D to find a clear frame. Enter, then drag a TIGHT box around ONE person.")

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
    return box


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


def format_hms(seconds: float) -> str:
    """Turn 95.2 into '1m 35s' so ETAs are easy to read."""
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    if minutes == 0:
        return f"{secs}s"
    return f"{minutes}m {secs}s"


def clip_frame_count(clip_path: Path) -> tuple[int, float]:
    """How many frames and what fps the clip has."""
    capture = cv2.VideoCapture(str(clip_path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
    capture.release()
    if frames <= 0:
        frames = int(TRACK_SECONDS * fps)
    return frames, fps


def find_beep_for_video(video_path: Path) -> float | None:
    """Extract audio and return the start-beep time in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        audio_start = time.perf_counter()
        extract_audio(video_path, wav_path)
        log_time("Extract audio", audio_start)
        beep_start = time.perf_counter()
        beep_time = find_beep(wav_path)
        log_time("Find beep", beep_start)
    return beep_time


def build_motion_signal(
    samples: list[tuple[float, float, float] | None],
    fps: float,
    beep_time: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Build a dense 0.01s motion curve after the beep.

    Returns (dense_times, dense_signal_px, body_scale) or None.
    """
    if fps <= 0 or not samples:
        return None

    times = []
    centers_x = []
    centers_y = []
    feet_y = []
    for i, sample in enumerate(samples):
        if sample is None:
            continue
        times.append(i / fps)
        centers_x.append(sample[0])
        centers_y.append(sample[1])
        feet_y.append(sample[2])

    if len(times) < 4:
        return None

    times_a = np.asarray(times, dtype=np.float64)
    cx_a = np.asarray(centers_x, dtype=np.float64)
    cy_a = np.asarray(centers_y, dtype=np.float64)
    fy_a = np.asarray(feet_y, dtype=np.float64)

    still = times_a < beep_time
    if int(np.count_nonzero(still)) < 3:
        still = times_a <= (times_a[0] + 0.5)
    if int(np.count_nonzero(still)) < 2:
        return None

    base_x = float(np.median(cx_a[still]))
    base_y = float(np.median(cy_a[still]))
    base_foot = float(np.median(fy_a[still]))
    body_scale = max(float(np.median(np.abs(fy_a[still] - cy_a[still]))) * 2.0, 40.0)

    frame_dist = np.sqrt((cx_a - base_x) ** 2 + (cy_a - base_y) ** 2)
    foot_dist = np.abs(fy_a - base_foot)
    frame_signal = np.maximum(frame_dist, foot_dist)

    end_t = float(times_a[-1])
    if end_t <= beep_time + RT_STEP_SECONDS:
        return None
    dense_t = np.arange(beep_time, end_t + 1e-9, RT_STEP_SECONDS)
    dense_signal = np.interp(dense_t, times_a, frame_signal)
    return dense_t, dense_signal, body_scale


def reaction_from_signal(
    dense_t: np.ndarray,
    dense_signal: np.ndarray,
    beep_time: float,
    threshold_px: float,
) -> tuple[float | None, float | None]:
    """First time the motion signal stays above threshold_px."""
    hold_needed = max(1, int(round(LEAVE_HOLD_SECONDS / RT_STEP_SECONDS)))
    hold = 0
    for idx, value in enumerate(dense_signal):
        if value >= threshold_px:
            hold += 1
            if hold >= hold_needed:
                move_index = idx - (hold_needed - 1)
                move_time = float(dense_t[move_index])
                return move_time, move_time - beep_time
        else:
            hold = 0
    return None, None


def leave_threshold_px(body_scale: float, body_frac: float | None = None) -> float:
    """Pixels for leave-block, scaled to how big the swimmer is in this clip."""
    frac = LEAVE_BLOCK_BODY_FRAC if body_frac is None else body_frac
    return max(8.0, float(frac) * body_scale)


def find_reaction_time(
    samples: list[tuple[float, float, float] | None],
    fps: float,
    beep_time: float,
    body_frac: float | None = None,
) -> tuple[float | None, float | None]:
    """
    Approximate World Aquatics reaction time from video.

    Official RT (Omega / World Aquatics):
      start signal -> swimmer's feet leave the block (contact switch opens),
      reported to 0.01s. That is NOT the first tiny muscle twitch.

    Our video estimate:
      1. Baseline = still pose on the block before the beep (center + feet).
      2. After the beep, build a motion signal every RT_STEP_SECONDS (0.01s)
         by interpolating between video frames.
      3. RT = first time motion exceeds LEAVE_BLOCK_BODY_FRAC * body size
         and stays there for LEAVE_HOLD_SECONDS.
    """
    built = build_motion_signal(samples, fps, beep_time)
    if built is None:
        return None, None
    dense_t, dense_signal, body_scale = built
    threshold_px = leave_threshold_px(body_scale, body_frac)
    return reaction_from_signal(dense_t, dense_signal, beep_time, threshold_px)


def calibrate_leave_block_frac(
    samples: list[tuple[float, float, float] | None],
    fps: float,
    beep_time: float,
    known_rt: float,
) -> float | None:
    """
    Optional one-time tune: learn LEAVE_BLOCK_BODY_FRAC from an official RT.

    Stores a fraction of body size (not pixels), so other zooms/distances
    can reuse the same constant without re-calibrating every video.
    """
    built = build_motion_signal(samples, fps, beep_time)
    if built is None:
        return None
    dense_t, dense_signal, body_scale = built

    leave_t = beep_time + known_rt
    if leave_t < float(dense_t[0]) or leave_t > float(dense_t[-1]):
        print(
            f"Known leave time {leave_t:.2f}s is outside the tracked clip "
            f"({dense_t[0]:.2f}s .. {dense_t[-1]:.2f}s)."
        )
        return None

    signal_at_leave = float(np.interp(leave_t, dense_t, dense_signal))
    frac_at_leave = signal_at_leave / body_scale

    best_frac = frac_at_leave
    best_err = abs(known_rt)
    for frac in np.linspace(max(0.05, frac_at_leave * 0.5), frac_at_leave * 1.5 + 0.05, 80):
        thr = leave_threshold_px(body_scale, float(frac))
        _move, rt = reaction_from_signal(dense_t, dense_signal, beep_time, thr)
        if rt is None:
            continue
        err = abs(rt - known_rt)
        if err < best_err:
            best_err = err
            best_frac = float(frac)

    print(f"Official RT you gave: {known_rt:.2f}s")
    print(f"Beep + official RT = leave at {leave_t:.2f}s in the video")
    print(
        f"Motion at that moment: {signal_at_leave:.1f}px "
        f"= {frac_at_leave:.2f} x body size ({body_scale:.1f}px)"
    )
    print(
        f"Best matching LEAVE_BLOCK_BODY_FRAC: {best_frac:.2f} "
        f"(reproduces RT within {best_err:.2f}s)"
    )
    print(
        f"Set this once near the top of main.py (then skip --known-rt):\n"
        f"LEAVE_BLOCK_BODY_FRAC = {best_frac:.2f}"
    )
    return best_frac


def vlm_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SWIM_VLM_API_KEY")
    return key.strip() if key else None


def encode_image_jpeg_b64(bgr: np.ndarray, max_side: int = VLM_JPEG_MAX_SIDE) -> str:
    """Resize and JPEG-encode a crop for the vision LM."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Could not encode crop as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_image_file_b64(path: Path, max_side: int = VLM_JPEG_MAX_SIDE) -> str | None:
    """Load a reference image from disk and JPEG-encode it for the LM."""
    if not path.is_file():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return encode_image_jpeg_b64(img, max_side=max_side)


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
) -> np.ndarray | None:
    """Pick ONE red platform contour (not lane ropes / skin blobs)."""
    h, w = red.shape[:2]
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(120.0, 0.006 * h * w)
    best = None
    best_score = -1e18
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
        cy = y + bh * 0.5
        # Prefer left half (blocks), not mid-pool.
        if cx > 0.72 * w:
            continue
        dist = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
        score = area - 1.8 * dist * dist
        if score > best_score:
            best_score = score
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
) -> tuple[int, np.ndarray, np.ndarray | None]:
    """
    Count skin/foot pixels on the BLOCK edge / just above it.

    Returns (overlap_pixels, red_mask, primary_contour_or_None).
    Empty red platform alone must score ~0 (no foot contact).
    """
    red = red_block_mask(bgr)
    cnt = primary_block_contour(red, anchor_xy=anchor_xy)
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


def annotate_starting_block(
    crop_bgr: np.ndarray,
    anchor_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Outline ONE primary starting BLOCK (yellow) for the vision LM.

    Prefer the block nearest the tracked swimmer's feet (anchor_xy).
    """
    out = crop_bgr.copy()
    h, w = out.shape[:2]
    red = red_block_mask(out)
    cnt = primary_block_contour(red, anchor_xy=anchor_xy)
    if cnt is not None:
        cv2.drawContours(out, [cnt], -1, (0, 255, 255), 2)
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.putText(
            out,
            "BLOCK",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    # Skip legend on tiny crops — it was covering the whole image.
    if h >= 220 and w >= 220:
        cv2.putText(
            out,
            "ON = foot on yellow BLOCK top",
            (8, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "LEFT = yellow BLOCK empty",
            (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


_VLM_REF_CACHE: dict[str, str] | None = None


def vlm_reference_b64() -> dict[str, str]:
    """Cached base64 for few-shot ON / LEFT example photos (clean, no overlays)."""
    global _VLM_REF_CACHE
    if _VLM_REF_CACHE is not None:
        return _VLM_REF_CACHE
    refs: dict[str, str] = {}
    for key, path in (
        ("ON_BLOCK", VLM_REF_ON),
        ("ON_BLOCK_HARD", VLM_REF_ON_HARD),
        ("ON_BLOCK_WEDGE", VLM_REF_ON_WEDGE),
        ("LEFT_BLOCK", VLM_REF_LEFT),
        ("LEFT_BLOCK_HARD", VLM_REF_LEFT_HARD),
    ):
        if not path.is_file():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        refs[key] = encode_image_jpeg_b64(img)
    _VLM_REF_CACHE = refs
    return refs


def vlm_ssl_context() -> ssl.SSLContext:
    """Use certifi CA bundle when present (fixes macOS CERTIFICATE_VERIFY_FAILED)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def parse_json_bool(value: object, default: bool = False) -> bool:
    """Parse JSON bools safely (bool('false') is True in Python — never use that)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", "none"):
            return False
    return default


def _foot_on(value: object) -> bool:
    """True if a foot field means still contacting the block."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("on", "contact", "touching", "yes", "true", "1", "rear", "front"):
        return True
    if s in ("off", "clear", "none", "no", "false", "0", "air", "water"):
        return False
    return False


def label_from_vlm_json(obj: dict) -> str:
    """Map structured LM evidence to ON_BLOCK / LEFT_BLOCK.

    Leave only when BOTH feet are clear. Any toe still touching => ON.
    Ambiguous answers default to ON (early LEFT underestimates RT).
    """
    rear = obj.get("rear_foot", obj.get("left_foot", obj.get("foot")))
    front = obj.get("front_foot", obj.get("right_foot"))
    any_toe = obj.get("any_toe_contact", obj.get("contact"))
    both_clear = obj.get("both_feet_clear")
    platform_empty = obj.get("platform_empty")
    gap = str(obj.get("gap", "")).strip().lower()

    rear_on = _foot_on(rear) if rear is not None else None
    front_on = _foot_on(front) if front is not None else None

    if rear_on is True or front_on is True:
        return "ON_BLOCK"
    if parse_json_bool(any_toe, default=False):
        return "ON_BLOCK"
    if both_clear is not None and not parse_json_bool(both_clear, False):
        return "ON_BLOCK"
    if gap in ("none", "tiny"):
        return "ON_BLOCK"

    if rear_on is False and front_on is False:
        return "LEFT_BLOCK"
    if parse_json_bool(both_clear, False):
        return "LEFT_BLOCK"
    if parse_json_bool(platform_empty, False) and not parse_json_bool(any_toe, False):
        return "LEFT_BLOCK"
    if gap == "clear" and rear_on is not True and front_on is not True:
        return "LEFT_BLOCK"

    foot = str(obj.get("foot", "")).strip().lower()
    if foot in ("rear", "front", "on"):
        return "ON_BLOCK"
    if foot in ("none", "off", "in_water") and parse_json_bool(any_toe, False) is False:
        if parse_json_bool(platform_empty, False) or gap == "clear":
            return "LEFT_BLOCK"

    return "ON_BLOCK"


def vlm_token_limit_kwargs(n: int = 120) -> dict[str, int]:
    """
    Newer OpenAI models (gpt-5*, o-series) reject max_tokens and require
    max_completion_tokens. Older vision models still want max_tokens.
    """
    name = VLM_MODEL.lower()
    if name.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}


def ask_vlm_on_or_left(crop_bgr: np.ndarray) -> tuple[str | None, str]:
    """
    Ask a vision LM for structured foot-vs-block evidence.

    Returns (label, raw_reply). Raises RuntimeError('vlm_auth') on HTTP 401.
    """
    key = vlm_api_key()
    if not key:
        return None, ""

    b64 = encode_image_jpeg_b64(crop_bgr)
    refs = vlm_reference_b64()

    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Side-view crop of ONE swim start. Decide if the MAIN swimmer "
                "still has ANY foot contact with the starting block "
                "(flat red top OR rear red wedge).\n\n"
                "Balanced rules:\n"
                "- ON if rear foot is still on the rear WEDGE, even when the "
                "front foot has already lifted and the body is diving.\n"
                "- ON if front toes are still pressed on the flat red top "
                "(no air under the contact point).\n"
                "- LEFT only when BOTH feet are clear of top AND wedge "
                "(visible gap under the last foot, or platform empty).\n"
                "- Front foot in the air alone does NOT mean LEFT while the "
                "rear foot remains on the wedge.\n"
                "- Foot past the front edge with a clear gap under the toes, "
                "AND rear foot also clear = LEFT.\n"
                "- Hands do not count. Ignore other lanes.\n\n"
                "Reply JSON only:\n"
                '{"rear_foot":"on"|"off","front_foot":"on"|"off",'
                '"any_toe_contact":true|false,"both_feet_clear":true|false,'
                '"gap":"none"|"tiny"|"clear"}'
            ),
        }
    ]
    if "ON_BLOCK" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": "EXAMPLE → ON: foot still contacting red top/wedge.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['ON_BLOCK']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "ON_BLOCK_WEDGE" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → ON (rear wedge): front foot may already be in "
                        "the air, but rear foot is STILL on the slanted red wedge. "
                        "rear_foot=on, both_feet_clear=false. This is ON, not LEFT."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['ON_BLOCK_WEDGE']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "ON_BLOCK_HARD" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → ON (last toes): diving, one leg up, but front "
                        "toes still pressed on the front red edge with no gap."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['ON_BLOCK_HARD']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "LEFT_BLOCK" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → LEFT: both feet clear / empty platform. "
                        "rear_foot=off, front_foot=off, both_feet_clear=true."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['LEFT_BLOCK']}",
                        "detail": "high",
                    },
                },
            ]
        )
    if "LEFT_BLOCK_HARD" in refs:
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "EXAMPLE → LEFT (both clear): diving with the front foot "
                        "near the edge BUT airborne with a gap, AND the rear foot "
                        "also off the wedge. Only then LEFT."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{refs['LEFT_BLOCK_HARD']}",
                        "detail": "high",
                    },
                },
            ]
        )
    user_content.extend(
        [
            {
                "type": "text",
                "text": (
                    "QUERY: Check the MAIN swimmer's REAR foot on the wedge AND "
                    "FRONT foot on the red top. If EITHER still touches, answer ON. "
                    "LEFT only if BOTH are clear. JSON only."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
    )


    payload = {
        "model": VLM_MODEL,
        "temperature": 0,
        **vlm_token_limit_kwargs(120),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a World Aquatics-style swim-start timer. Output JSON only. "
                    "ON if either foot still contacts the red top OR rear wedge. "
                    "A lifted front foot with rear foot still on the wedge is ON. "
                    "LEFT only when both feet are clear. Hands do not count."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    }
    req = urllib.request.Request(
        f"{VLM_API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90, context=vlm_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code == 401:
            raise RuntimeError("vlm_auth") from err
        detail = ""
        try:
            detail = err.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        print(f"  Vision LM request failed: HTTP {err.code} {err.reason} {detail}")
        return None, ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        print(f"  Vision LM request failed: {err}")
        return None, ""

    try:
        text_out = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError):
        return None, ""

    raw = text_out
    if "```" in raw:
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    start_j = raw.find("{")
    end_j = raw.rfind("}")
    if start_j < 0 or end_j < start_j:
        upper = text_out.upper()
        if "LEFT" in upper and "ON" not in upper:
            return "LEFT_BLOCK", text_out
        if "ON" in upper:
            return "ON_BLOCK", text_out
        return None, text_out
    try:
        obj = json.loads(raw[start_j : end_j + 1])
    except json.JSONDecodeError:
        return None, text_out
    if not isinstance(obj, dict):
        return None, text_out
    return label_from_vlm_json(obj), text_out



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


def extract_crop(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    xa, ya, xb, yb = rect
    return frame[ya:yb, xa:xb].copy()


def read_padded_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    pad: int | None = None,
) -> np.ndarray:
    """Backward-compatible helper: one stable crop from the track box."""
    del pad
    return extract_crop(frame, crop_rect_from_box(frame.shape, box))



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


def smooth_vlm_isolated_on(
    labels: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """
    Treat a single ON between two LEFT answers as LM flicker (keep LEFT).

    Real 'still on' usually shows as several ON frames, not one blip after leave.
    """
    if len(labels) < 3:
        return labels
    out = list(labels)
    for i in range(1, len(out) - 1):
        if (
            out[i][1] == "ON_BLOCK"
            and out[i - 1][1] == "LEFT_BLOCK"
            and out[i + 1][1] == "LEFT_BLOCK"
        ):
            out[i] = (out[i][0], "LEFT_BLOCK")
    return out


def confirmed_vlm_leave_time(labels: list[tuple[float, str]]) -> float | None:
    """
    Leave = first of VLM_LEFT_CONFIRM LEFT answers in a row AFTER the last ON.

    Early false LEFT is ignored if a later frame is still ON_BLOCK.
    A lone ON between LEFTs is smoothed away first (LM flicker).
    """
    labels = smooth_vlm_isolated_on(labels)
    need = max(1, int(VLM_LEFT_CONFIRM))
    last_on = -1
    for i, (_t, label) in enumerate(labels):
        if label == "ON_BLOCK":
            last_on = i

    start = last_on + 1
    for i in range(start, len(labels)):
        if labels[i][1] != "LEFT_BLOCK":
            continue
        streak = 0
        for j in range(i, len(labels)):
            if labels[j][1] != "LEFT_BLOCK":
                break
            streak += 1
            if streak >= need:
                return float(labels[i][0])
    return None


def vlm_sample_times(beep_time: float, end_t: float) -> np.ndarray:
    """
    Absolute sample times for LM crops after the beep.

    Full window: [beep+0.50, beep+1.20] (clamped by end_t).
    Dense 0.03s steps in [beep+0.58, beep+0.80]; 0.05s elsewhere.
    """
    start_t = beep_time + VLM_RT_MIN_SECONDS
    if start_t >= end_t:
        return np.asarray([], dtype=float)

    dense_lo = beep_time + VLM_DENSE_MIN_SECONDS
    dense_hi = min(beep_time + VLM_DENSE_MAX_SECONDS, end_t)
    times: list[float] = []

    # Coarse before dense band: 0.50 .. just before 0.58
    t = start_t
    while t < dense_lo - 1e-9 and t <= end_t + 1e-9:
        times.append(float(t))
        t += VLM_SAMPLE_COARSE_STEP_SECONDS

    # Dense band: 0.58 .. 0.80
    if dense_lo <= end_t + 1e-9:
        t = max(dense_lo, start_t)
        while t <= dense_hi + 1e-9:
            times.append(float(t))
            t += VLM_SAMPLE_DENSE_STEP_SECONDS

    # Coarse after dense band: after 0.80 .. 1.20
    t = beep_time + VLM_DENSE_MAX_SECONDS + VLM_SAMPLE_COARSE_STEP_SECONDS
    while t <= end_t + 1e-9:
        times.append(float(t))
        t += VLM_SAMPLE_COARSE_STEP_SECONDS

    if not times:
        return np.asarray([], dtype=float)
    # Deduplicate / sort (float drift); keep unique within 1ms.
    arr = np.asarray(sorted(times), dtype=float)
    keep = [True]
    for i in range(1, len(arr)):
        keep.append(arr[i] - arr[i - 1] > 0.001)
    return arr[np.asarray(keep)]


def find_leave_with_vlm(
    clip_path: Path,
    track_boxes: list[tuple[int, int, int, int] | None],
    fps: float,
    beep_time: float,
    crop_dir: Path | None = None,
) -> float | None:
    """
    Leave-block time from a vision LM.

    Samples crops from beep+0.50 to beep+1.20: every 0.03s in 0.58–0.80,
    every 0.05s elsewhere. Accepts leave only after VLM_LEFT_CONFIRM LEFT
    answers in a row.
    """
    if vlm_api_key() is None:
        print(
            "No OPENAI_API_KEY (or SWIM_VLM_API_KEY) set — "
            "skipping vision LM, using motion only."
        )
        return None
    if fps <= 0 or not track_boxes:
        return None

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        print("Could not open clip for vision LM crops.")
        return None

    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
        for old in crop_dir.glob("*.jpg"):
            old.unlink()

    # Reload clean few-shot refs (no yellow overlays from older runs).
    global _VLM_REF_CACHE
    _VLM_REF_CACHE = None

    end_t = min(beep_time + VLM_RT_SEARCH_SECONDS, (len(track_boxes) - 1) / fps)
    start_t = beep_time + VLM_RT_MIN_SECONDS
    if start_t >= end_t:
        print(
            f"Vision LM: search window empty "
            f"(start {start_t:.2f}s >= end {end_t:.2f}s)."
        )
        capture.release()
        return None
    sample_times = vlm_sample_times(beep_time, end_t)
    if sample_times.size == 0:
        capture.release()
        return None
    print(
        f"Vision LM ({VLM_MODEL}): checking {len(sample_times)} crops "
        f"from {start_t:.2f}s to {end_t:.2f}s "
        f"(0.03s in {VLM_DENSE_MIN_SECONDS:.2f}–{VLM_DENSE_MAX_SECONDS:.2f}s after beep, "
        f"else 0.05s; need {VLM_LEFT_CONFIRM} LEFT in a row)..."
    )
    if crop_dir is not None:
        print(f"Saving vision LM crops in: {crop_dir}")
    refs = vlm_reference_b64()
    if refs:
        print(
            f"Few-shot refs: {', '.join(refs.keys())} "
            f"from {VLM_REFS_DIR}"
        )
    else:
        print(f"Warning: no few-shot refs found in {VLM_REFS_DIR}")

    labels: list[tuple[float, str]] = []
    # Lock crop using the swimmer box AT THE BEEP (still on the block),
    # not at beep+0.5s when the track box may already be tiny mid-dive.
    locked_rect: tuple[int, int, int, int] | None = None
    beep_idx = int(round(beep_time * fps))
    seed_box = nearest_track_box(track_boxes, beep_idx)
    if seed_box is None:
        seed_box = nearest_track_box(track_boxes, max(0, beep_idx - 5))
    for t in sample_times:
        frame_index = int(round(t * fps))
        frame_index = min(max(frame_index, 0), len(track_boxes) - 1)
        box = nearest_track_box(track_boxes, frame_index)
        if box is None and seed_box is None:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        if locked_rect is None:
            lock_box = seed_box if seed_box is not None else box
            locked_rect = crop_rect_from_box(frame.shape, lock_box)
            lw = locked_rect[2] - locked_rect[0]
            lh = locked_rect[3] - locked_rect[1]
            print(
                f"Locked VLM crop window {locked_rect} "
                f"({lw}x{lh}px, from beep-time box)"
            )
        raw = extract_crop(frame, locked_rect)
        rh, rw = raw.shape[:2]
        # Anchor toward the left/lower part of the locked window (block side).
        anchor = (rw * 0.35, rh * 0.60)
        overlap_px, _red, cnt = foot_block_overlap_px(raw, anchor_xy=anchor)
        crop = raw  # clean crop — no yellow overlay; prompt + few-shot only
        # Always ask the LM. Local overlap is only a soft veto after a LEFT.
        try:
            label, raw_vlm = ask_vlm_on_or_left(crop)
        except RuntimeError as err:
            if str(err) == "vlm_auth":
                capture.release()
                print(
                    "Vision LM: HTTP 401 Unauthorized — API key rejected.\n"
                    "  Fix: create a new secret key at https://platform.openai.com/api-keys\n"
                    "  Put only this line in .env (no quotes/spaces):\n"
                    "  OPENAI_API_KEY=sk-...\n"
                    "  Falling back to motion-only RT."
                )
                return None
            raise
        detail = f"vlm+overlap={overlap_px}px"
        if raw_vlm:
            one_line = " ".join(raw_vlm.split())
            if len(one_line) > 120:
                one_line = one_line[:117] + "..."
            detail += f" | {one_line}"
        if (
            cnt is not None
            and label == "LEFT_BLOCK"
            and overlap_px >= VLM_OVERLAP_FORCE_ON_PX
        ):
            label = "ON_BLOCK"
            detail = f"veto_LEFT overlap={overlap_px}px"
        tag = label if label is not None else "NO_ANSWER"
        if crop_dir is not None:
            # Draw overlap count so you can audit crops.
            vis = crop.copy()
            cv2.putText(
                vis,
                detail,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            out_name = f"t{t:05.2f}s_{tag}.jpg".replace(":", "-")
            cv2.imwrite(str(crop_dir / out_name), vis)
        if label is None:
            print(f"  t={t:.2f}s  ? (no answer)  {detail}")
            continue
        print(f"  t={t:.2f}s  {label}  ({detail})")
        labels.append((float(t), label))

        # After a confirmed streak, take one extra LEFT then stop.
        leave = confirmed_vlm_leave_time(labels)
        if leave is not None:
            left_count = sum(1 for _, lab in labels if lab == "LEFT_BLOCK")
            if left_count >= VLM_LEFT_CONFIRM + 1:
                break

    capture.release()

    first_left = confirmed_vlm_leave_time(labels)
    if first_left is None:
        print(
            "Vision LM: no confirmed leave "
            f"({VLM_LEFT_CONFIRM} LEFT_BLOCK in a row). "
            "A single LEFT followed by ON_BLOCK is ignored."
        )
        return None

    print(
        f"Vision LM leave-block at {first_left:.2f}s "
        f"(confirmed by {VLM_LEFT_CONFIRM} LEFT in a row)"
    )
    return first_left


def refine_reaction_with_vlm_anchor(
    samples: list[tuple[float, float, float] | None],
    fps: float,
    beep_time: float,
    vlm_leave_time: float,
) -> tuple[float | None, float | None]:
    """
    Fine RT on top of the LM: use motion size at the LM leave moment as
    the scale, then find the earliest crossing after the beep.
    """
    built = build_motion_signal(samples, fps, beep_time)
    if built is None:
        return None, None
    dense_t, dense_signal, _body_scale = built

    if vlm_leave_time < float(dense_t[0]) or vlm_leave_time > float(dense_t[-1]):
        return None, None

    signal_at_vlm = float(np.interp(vlm_leave_time, dense_t, dense_signal))
    early = dense_t <= (beep_time + 0.15)
    noise = float(np.median(dense_signal[early])) if np.any(early) else 0.0
    threshold = max(noise + 8.0, VLM_REFINE_SIGNAL_FRAC * signal_at_vlm, 12.0)

    move_time, reaction = reaction_from_signal(
        dense_t, dense_signal, beep_time, threshold
    )
    if move_time is None:
        # Fallback: trust the LM time if motion never crossed cleanly.
        return vlm_leave_time, vlm_leave_time - beep_time

    # Keep refine near the LM answer (don't accept a tiny early twitch).
    if abs(move_time - vlm_leave_time) > 0.35:
        print(
            f"Motion refine {move_time:.2f}s was far from LM {vlm_leave_time:.2f}s — "
            "preferring LM time with light pull-back."
        )
        # Pull slightly earlier than LM using local signal rise.
        window = (dense_t >= vlm_leave_time - 0.20) & (dense_t <= vlm_leave_time)
        if np.any(window):
            local_t = dense_t[window]
            local_s = dense_signal[window]
            local_thr = max(noise + 8.0, 0.40 * signal_at_vlm)
            for t, s in zip(local_t, local_s):
                if s >= local_thr:
                    return float(t), float(t) - beep_time
        return vlm_leave_time, vlm_leave_time - beep_time

    print(
        f"Motion refine: threshold {threshold:.1f}px "
        f"(LM motion was {signal_at_vlm:.1f}px, noise {noise:.1f}px)"
    )
    return move_time, reaction


def find_reaction_time_hybrid(
    clip_path: Path,
    track_boxes: list[tuple[int, int, int, int] | None],
    samples: list[tuple[float, float, float] | None],
    fps: float,
    beep_time: float,
    use_vlm: bool,
    body_frac: float | None = None,
    crop_dir: Path | None = None,
) -> tuple[float | None, float | None, str]:
    """
    Primary RT = vision-LM leave time. Also compute motion refine and print it
    on a separate line. Fall back to body-frac motion if LM is unavailable.

    Returns (move_time, reaction, method_label) for the primary (LM) estimate.
    """
    if use_vlm:
        vlm_leave = find_leave_with_vlm(
            clip_path, track_boxes, fps, beep_time, crop_dir=crop_dir
        )
        if vlm_leave is not None:
            lm_rt = vlm_leave - beep_time
            print(
                f"Vision LM RT: {lm_rt:.2f}s "
                f"(leave at {vlm_leave:.2f}s)"
            )
            refine_leave, refine_rt = refine_reaction_with_vlm_anchor(
                samples, fps, beep_time, vlm_leave
            )
            if refine_rt is not None and refine_leave is not None:
                print(
                    f"Motion refine RT: {refine_rt:.2f}s "
                    f"(leave at {refine_leave:.2f}s)"
                )
            else:
                print("Motion refine RT: unavailable")
            return vlm_leave, lm_rt, "vision-LM (primary); motion refine also printed"

    move_time, reaction = find_reaction_time(
        samples, fps, beep_time, body_frac=body_frac
    )
    return move_time, reaction, "motion only (body-frac)"


def burn_reaction_overlay(
    raw_path: Path,
    out_path: Path,
    fps: float,
    beep_time: float | None,
    move_time: float | None,
    reaction: float | None,
) -> None:
    """Add beep / move / reaction text onto the tracked video, then H.264 encode."""
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
            t = index / fps
            lines = [f"t={t:.2f}s"]
            if beep_time is not None:
                lines.append(f"beep {beep_time:.2f}s")
            if move_time is not None:
                lines.append(f"leave {move_time:.2f}s")
            if reaction is not None:
                lines.append(f"RT {reaction:.2f}s")
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
            # Green flash on the movement frame.
            if move_time is not None and abs(t - move_time) < (0.5 / fps):
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 0), 8)
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


def track_one_swimmer(
    video_path: Path,
    known_rt: float | None = None,
    use_vlm: bool = True,
) -> None:
    """
    Find the beep, follow one swimmer with SAM 2 for TRACK_SECONDS,
    then measure reaction time = leave-block after the beep.

    Prefer vision LM leave time when an API key is set; else body-frac motion.
    is set; otherwise fall back to body-frac motion.
    """
    print(f"Finding start beep...")
    beep_time = find_beep_for_video(video_path)
    if beep_time is None:
        print("No clear beep found. Tracking will still run, but RT needs a beep.")
    else:
        print(f"The beep occurred at {beep_time:.3f} seconds")

    print(f"Using only the first {TRACK_SECONDS} seconds...")
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / f"{video_path.stem}_first{TRACK_SECONDS}s.mp4"
        clip_first_seconds(video_path, clip_path, TRACK_SECONDS)

        box = click_swimmer(clip_path)
        if box is None:
            print("No box, skipping track.")
            return

        print(f"Tracking one swimmer inside box {box} with SAM 2...")
        print("First run downloads the SAM 2 file.")

        total_frames, video_fps = clip_frame_count(clip_path)
        print(
            f"Clip: {total_frames} frames, {video_fps:.1f} fps, "
            f"{total_frames / video_fps:.1f}s of video."
        )
        print(
            "SAM 2 runs a neural net on EVERY frame on your CPU, "
            "so 10s of video can take many minutes."
        )

        start = time.perf_counter()

        from ultralytics.models.sam import SAM2VideoPredictor

        OUTPUT_FOLDER.mkdir(exist_ok=True)
        out_dir = OUTPUT_FOLDER / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_path.stem}_track{TRACK_SECONDS}s.mp4"

        predictor = SAM2VideoPredictor(
            overrides={
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "imgsz": 640,
                "model": SAM_TRACK_MODEL,
                "save": False,
                "verbose": False,
            }
        )

        raw_path = Path(tmp) / "raw_track.mp4"
        writer = None
        frame_count = 0
        first_area = None
        last_good_mask = None
        # Where we think the swimmer is / is going (stops lane switches).
        pred_x = (box[0] + box[2]) / 2.0
        pred_y = (box[1] + box[3]) / 2.0
        vel_x = 0.0
        vel_y = 0.0
        # One (cx, cy, foot_y) per frame. None = no mask that frame.
        samples: list[tuple[float, float, float] | None] = []
        # Bounding box per frame for vision-LM crops (clean frames from clip).
        track_boxes: list[tuple[int, int, int, int] | None] = []
        results = predictor(
            source=str(clip_path),
            bboxes=box,
            stream=True,
        )
        for result in results:
            frame = result.orig_img.copy()
            height, width = frame.shape[:2]
            if writer is None:
                fps = video_fps if video_fps > 0 else 30
                writer = cv2.VideoWriter(
                    str(raw_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            sample: tuple[float, float, float] | None = None
            frame_box: tuple[int, int, int, int] | None = None
            if result.masks is not None and len(result.masks) > 0:
                mask = result.masks.data[0].cpu().numpy()
                mask = cv2.resize(mask.astype(np.float32), (width, height))
                # Follow the moving person, not the old click box on the blocks.
                mask = keep_one_person_mask(
                    mask,
                    (pred_x, pred_y),
                    first_area,
                    MAX_CENTER_JUMP_PX,
                )
                if np.count_nonzero(mask) == 0 and last_good_mask is not None:
                    mask = last_good_mask
                area = float(np.count_nonzero(mask))
                if area > 0:
                    if first_area is None:
                        first_area = area
                    last_good_mask = mask
                    colored = frame.copy()
                    colored[mask > 0.5] = (0, 180, 255)
                    frame = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)
                    ys, xs = np.where(mask > 0.5)
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    frame_box = (x1, y1, x2, y2)
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    foot_y = float(ys.max())  # bottom of body ~ feet on the block
                    sample = (cx, cy, foot_y)
                    # Smooth velocity so the next frame prediction stays on THIS diver.
                    vel_x = 0.7 * vel_x + 0.3 * (cx - pred_x)
                    vel_y = 0.7 * vel_y + 0.3 * (cy - pred_y)
                    pred_x = cx + vel_x
                    pred_y = cy + vel_y
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (int(cx), int(foot_y)), 5, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                    cv2.putText(
                        frame,
                        "swimmer",
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 180, 255),
                        2,
                    )
            else:
                # No mask: keep coasting along the last velocity.
                pred_x += vel_x
                pred_y += vel_y
            samples.append(sample)
            track_boxes.append(frame_box)
            writer.write(frame)
            frame_count += 1
            elapsed = time.perf_counter() - start
            # After a few frames we know the real speed, then we can guess the rest.
            if frame_count in (3, 10) or frame_count % 15 == 0:
                per_frame = elapsed / frame_count
                remaining = per_frame * (total_frames - frame_count)
                print(
                    f"  {frame_count}/{total_frames} frames  "
                    f"({per_frame:.2f}s each)  "
                    f"elapsed {format_hms(elapsed)}  "
                    f"ETA {format_hms(remaining)}"
                )
                if frame_count == 3:
                    print(
                        f"  Estimated total time: "
                        f"{format_hms(per_frame * total_frames)}"
                    )

        if writer is not None:
            writer.release()
        log_time(f"SAM 2 track ({frame_count} frames)", start)

        move_time = None
        reaction = None
        calibrated_frac = None
        if beep_time is not None:
            print(
                "World Aquatics RT = beep -> feet leave the block "
                "(pressure switch), to 0.01s. Video approximates that."
            )
            if known_rt is not None:
                print(
                    f"One-time tune of LEAVE_BLOCK_BODY_FRAC "
                    f"from known RT={known_rt:.2f}s..."
                )
                print("Box the same swimmer who had that official RT.")
                calibrated_frac = calibrate_leave_block_frac(
                    samples, video_fps, beep_time, known_rt
                )

            move_time, reaction, method = find_reaction_time_hybrid(
                clip_path,
                track_boxes,
                samples,
                video_fps,
                beep_time,
                use_vlm=use_vlm and known_rt is None,
                body_frac=calibrated_frac,
                crop_dir=out_dir / "vlm_crops",
            )
            print(f"RT method: {method}")
            if reaction is None:
                print("Could not find a clear leave-block movement after the beep.")
            else:
                print(f"Leave-block (LM) at {move_time:.2f} seconds")
                print(
                    f"Reaction time (LM): {reaction:.2f} seconds "
                    f"({reaction * 1000:.0f} ms)"
                )
                if known_rt is not None:
                    print(
                        f"Official was {known_rt:.2f}s  |  "
                        f"video estimate {reaction:.2f}s  |  "
                        f"error {abs(reaction - known_rt):.2f}s"
                    )
                elif reaction < 0.50:
                    print(
                        "Warning: RT under 0.50s is rare for a dive start. "
                        "Check the tracked crop / vision LM labels."
                    )

        burn_reaction_overlay(
            raw_path,
            out_path,
            video_fps if video_fps > 0 else 30,
            beep_time,
            move_time,
            reaction,
        )

    print(f"Saved tracked video in: {out_path}")


def process_video(video_path: Path, models: dict) -> None:
    """Find the beep, then draw person boxes with YOLO."""
    print(f"Starting with video: {video_path}")
    video_start = time.perf_counter()

    # Temporary folder is deleted automatically when we leave this block.
    # We only need the .wav while find_beep() runs, then it can go away.
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        audio_start = time.perf_counter()
        extract_audio(video_path, wav_path)
        log_time("Extract audio", audio_start)

        beep_start = time.perf_counter()
        beep_time = find_beep(wav_path)
        log_time("Find beep", beep_start)

    if beep_time is None:
        print("No clear beep found in the first 15 seconds.")
    else:
        print(f"The beep occurred at {beep_time:.3f} seconds")

    detect_people(video_path, models)
    log_time("Video total", video_start)


def run(
    folder: str,
    video_name: str | None,
    track: bool,
    known_rt: float | None,
    use_vlm: bool,
) -> None:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise SystemExit(f"Folder not found: {folder_path}")

    run_start = time.perf_counter()
    videos = pick_video(folder_path, video_name)

    # Click-to-follow one swimmer. Skip YOLO entirely.
    if track:
        for video_path in videos:
            track_one_swimmer(video_path, known_rt=known_rt, use_vlm=use_vlm)
        log_time("All done", run_start)
        return

    # Load YOLO once, then reuse it for every video in the folder.
    models = load_yolo_model()
    for video_path in videos:
        process_video(video_path, models)
    log_time("All done", run_start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Swim Race AI")

    # -f is the folder that already contains your videos.
    parser.add_argument(
        "-f",
        "--folder",
        required=True,
        help="Path to the folder that contains your videos",
    )
    # -v is optional. Leave it out to check every video in the folder.
    parser.add_argument(
        "-v",
        "--video",
        help="Optional: one video filename inside that folder",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Click one swimmer: SAM 2 tracks and measures reaction time after the beep",
    )
    parser.add_argument(
        "--known-rt",
        type=float,
        default=None,
        help="Optional: official RT (e.g. 0.62) to tune LEAVE_BLOCK_BODY_FRAC once",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip vision LM; use motion-only reaction time",
    )
    args = parser.parse_args()
    run(args.folder, args.video, args.track, args.known_rt, use_vlm=not args.no_vlm)


# This special check is True only when you run this file directly:
#   python main.py -f videos
# It is False if another file does: import main
# That way importing this file will not accidentally start the program.
if __name__ == "__main__":
    main()
