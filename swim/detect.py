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
from .cuts import detect_cuts
from .lanes import lane_band_y_range, lane_from_center_y, pick_best_for_lane


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


def load_swimmer_yolov5(weights: Path | None = None):
    """
    Load DBDoco's YOLOv5 ``person_swimmer`` weights.

    These are classic Ultralytics YOLOv5 checkpoints (not YOLO v8+), so they
    must be loaded through ``third_party/yolov5``. The weights were pickled on
    Windows; remapping ``WindowsPath`` → ``PosixPath`` is required on macOS.
    """
    import pathlib
    import sys

    import torch

    weights_path = (weights or YOLO_SWIMMER_MODEL).resolve()
    repo_path = YOLOV5_REPO.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Swimmer YOLO weights missing: {weights_path}\n"
            "Download models/exp5/weights/best.pt from "
            "https://github.com/DBDoco/yolo-swimmer-detection"
        )
    if not (repo_path / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"YOLOv5 code missing at {repo_path}. Clone with:\n"
            "  git clone --depth 1 https://github.com/ultralytics/yolov5.git "
            "third_party/yolov5"
        )

    # Checkpoint contains WindowsPath objects from the original training host.
    pathlib.WindowsPath = pathlib.PosixPath
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from models.common import AutoShape, DetectMultiBackend
    from utils.torch_utils import select_device

    device = select_device("cpu")
    backend = DetectMultiBackend(str(weights_path), device=device, fuse=True)
    model = AutoShape(backend)
    # AutoShape defaults to conf=0.25; keep our preview threshold explicit.
    model.conf = YOLO_SWIMMER_CONFIDENCE
    model.iou = YOLO_IOU
    return model


def _yolov5_boxes(model, frame: np.ndarray) -> list[tuple[list[int], float, str]]:
    """Run one BGR frame through the YOLOv5 AutoShape model."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model(rgb, size=YOLO_SWIMMER_IMAGE_SIZE)
    boxes: list[tuple[list[int], float, str]] = []
    if result.xyxy is None or len(result.xyxy) == 0:
        return boxes
    tensor = result.xyxy[0]
    if tensor is None or len(tensor) == 0:
        return boxes
    names = getattr(model, "names", {}) or {}
    for row in tensor.cpu().tolist():
        x1, y1, x2, y2, confidence, class_id = row[:6]
        label = names.get(int(class_id), f"class{int(class_id)}")
        boxes.append(
            ([int(x1), int(y1), int(x2), int(y2)], float(confidence), str(label))
        )
    return boxes


def preview_swimmers_after_first_cut(
    video_path: Path,
    preview_seconds: float = YOLO_SWIMMER_PREVIEW_SECONDS,
    lane: int | None = None,
) -> Path | None:
    """
    Draw specialized-swimmer YOLO detections after the first camera cut.

    Without ``lane``: raw evaluation (all boxes, including false positives).
    With ``lane``: tint that lane band, gray out other detections, and highlight
    the best box whose center falls in the requested lane
    (8 = closest/bottom, 1 = furthest/top).
    """
    if preview_seconds <= 0:
        raise ValueError("preview_seconds must be greater than zero")
    if lane is not None and not 1 <= lane <= LANE_COUNT:
        raise ValueError(f"lane must be in 1..{LANE_COUNT}")

    print(
        f"Finding the first camera cut inside the first "
        f"{preview_seconds:.1f}s..."
    )
    cuts = detect_cuts(video_path, max_seconds=preview_seconds)
    if not cuts:
        print("No hard cut found in the preview window; YOLO preview skipped.")
        return None
    first_cut = cuts[0]
    print(
        f"First cut: {first_cut.time_seconds:.3f}s "
        f"(frame {first_cut.frame_index})."
    )
    if lane is not None:
        print(
            f"Lane filter ON: lane {lane} "
            f"({'closest/bottom' if lane == LANE_COUNT else 'furthest/top' if lane == 1 else 'mid'})."
        )

    print(
        f"Loading specialized swimmer YOLO "
        f"({YOLO_SWIMMER_MODEL.name}, class person_swimmer)..."
    )
    model = load_swimmer_yolov5()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    preview_frames = min(
        max(1, int(round(preview_seconds * fps))),
        total_source_frames if total_source_frames > 0 else 2**31 - 1,
    )

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    out_dir = OUTPUT_FOLDER / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds_label = f"{preview_seconds:g}".replace(".", "p")
    lane_tag = f"_lane{lane}" if lane is not None else ""
    out_path = out_dir / (
        f"{video_path.stem}_yolo_swimmers_after_cut"
        f"{lane_tag}_{seconds_label}s.mp4"
    )

    band = None
    if lane is not None:
        band = lane_band_y_range(
            lane,
            height,
            LANE_COUNT,
            near_is_bottom=SAM3_NEAR_LANE_IS_BOTTOM,
            pool_y_top_frac=SAM3_POOL_Y_TOP_FRAC,
            pool_y_bottom_frac=SAM3_POOL_Y_BOTTOM_FRAC,
        )

    max_swimmers = 0
    detection_frames = 0
    lane_hits = 0
    last_boxes: list[tuple[list[int], float, str]] = []
    chosen: tuple[tuple[int, int, int, int], float] | None = None
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw_yolo_swimmers.mp4"
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create temporary YOLO preview video")

        frame_index = 0
        while frame_index < preview_frames:
            ok, frame = capture.read()
            if not ok:
                break
            absolute_time = frame_index / fps

            if frame_index >= first_cut.frame_index:
                relative_index = frame_index - first_cut.frame_index
                should_detect = (
                    relative_index % YOLO_SWIMMER_EVERY_N_FRAMES == 0
                )
                if should_detect:
                    last_boxes = _yolov5_boxes(model, frame)
                    detection_frames += 1
                    max_swimmers = max(max_swimmers, len(last_boxes))
                    if lane is not None:
                        plain = [
                            ((x1, y1, x2, y2), conf)
                            for (x1, y1, x2, y2), conf, _label in last_boxes
                        ]
                        picked = pick_best_for_lane(
                            plain,
                            lane,
                            height,
                            LANE_COUNT,
                            near_is_bottom=SAM3_NEAR_LANE_IS_BOTTOM,
                            pool_y_top_frac=SAM3_POOL_Y_TOP_FRAC,
                            pool_y_bottom_frac=SAM3_POOL_Y_BOTTOM_FRAC,
                        )
                        if picked is not None:
                            chosen = picked
                            lane_hits += 1

                if band is not None:
                    band_y0, band_y1 = band
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (0, band_y0),
                        (width - 1, band_y1),
                        (0, 180, 255),
                        -1,
                    )
                    frame = cv2.addWeighted(frame, 0.82, overlay, 0.18, 0)
                    cv2.rectangle(
                        frame,
                        (0, band_y0),
                        (width - 1, band_y1),
                        (0, 180, 255),
                        2,
                    )

                for (x1, y1, x2, y2), confidence, label in last_boxes:
                    if lane is None:
                        color = (255, 0, 255)
                        text = f"{label} {confidence:.2f}"
                    else:
                        assigned = lane_from_center_y(
                            0.5 * (y1 + y2),
                            height,
                            LANE_COUNT,
                            near_is_bottom=SAM3_NEAR_LANE_IS_BOTTOM,
                            pool_y_top_frac=SAM3_POOL_Y_TOP_FRAC,
                            pool_y_bottom_frac=SAM3_POOL_Y_BOTTOM_FRAC,
                        )
                        if assigned == lane:
                            color = (255, 0, 255)
                            text = f"L{assigned} {confidence:.2f}"
                        elif assigned is None:
                            color = (80, 80, 80)
                            text = f"? {confidence:.2f}"
                        else:
                            color = (160, 160, 160)
                            text = f"L{assigned} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        text,
                        (x1, max(y1 - 9, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                if chosen is not None:
                    (x1, y1, x2, y2), confidence = chosen
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                    cv2.putText(
                        frame,
                        f"LANE {lane} pick {confidence:.2f}",
                        (x1, min(y2 + 28, height - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if lane is None:
                    status = (
                        f"YOLO person_swimmer | t={absolute_time:.2f}s | "
                        f"detections={len(last_boxes)}"
                    )
                else:
                    status = (
                        f"YOLO lane {lane} | t={absolute_time:.2f}s | "
                        f"raw={len(last_boxes)} | lane_hits={lane_hits}"
                    )
            else:
                waiting = (
                    f"Waiting for first cut at {first_cut.time_seconds:.3f}s | "
                    f"t={absolute_time:.2f}s"
                )
                if lane is not None:
                    waiting += f" | target lane {lane}"
                status = waiting

            cv2.rectangle(frame, (0, 0), (width, 44), (0, 0, 0), -1)
            cv2.putText(
                frame,
                status,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1

            if frame_index % max(int(round(fps * 5)), 1) == 0:
                elapsed = time.perf_counter() - start
                remaining = (
                    elapsed / frame_index * (preview_frames - frame_index)
                )
                print(
                    f"  {frame_index}/{preview_frames} frames, "
                    f"elapsed {format_hms(elapsed)}, "
                    f"ETA {format_hms(remaining)}"
                )

        capture.release()
        writer.release()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-t",
                f"{frame_index / fps:.6f}",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )

    log_time(
        f"YOLO swimmer preview ({frame_index} frames; "
        f"{detection_frames} inference frames)",
        start,
    )
    print(f"Most raw swimmer boxes in one frame: {max_swimmers}")
    if lane is not None:
        if lane_hits == 0:
            print(f"No YOLO boxes fell into lane {lane}.")
        else:
            print(f"Lane {lane} hit on {lane_hits} inference frames.")
    print(f"Saved YOLO swimmer preview: {out_path}")
    return out_path



