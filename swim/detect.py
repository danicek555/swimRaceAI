"""swimRaceAI — detect."""

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
from ultralytics import YOLO

try:
    from ultralytics import YOLOWorld
except ImportError:
    YOLOWorld = None

from .config import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .blocks import *  # noqa: F401,F403


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



