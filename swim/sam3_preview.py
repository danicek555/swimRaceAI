"""SAM 3 concept segmentation after the first camera cut, filtered by lane.

Experiment mode, not production identity tracking:

1. Detect the first hard camera cut.
2. Prompt SAM 3 with text concepts such as ``swimmer``.
3. Assign each returned box/mask to a lane band
   (lane 8 = closest / bottom, lane 1 = furthest / top).
4. Highlight the requested lane's best detection.

SAM 3 weights (``sam3.pt``) are gated on Hugging Face and are NOT auto-
downloaded by Ultralytics. Place ``sam3.pt`` in the project root after
requesting access, then run ``--sam3-after-cut --lane N``.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .config import (
    LANE_COUNT,
    OUTPUT_FOLDER,
    SAM3_AFTER_CUT_SECONDS,
    SAM3_CONFIDENCE,
    SAM3_EVERY_N_FRAMES,
    SAM3_IMAGE_SIZE,
    SAM3_MODEL,
    SAM3_NEAR_LANE_IS_BOTTOM,
    SAM3_POOL_Y_BOTTOM_FRAC,
    SAM3_POOL_Y_TOP_FRAC,
    SAM3_TEXT_PROMPTS,
)
from .cuts import detect_cuts
from .lanes import lane_band_y_range, lane_from_center_y, pick_best_for_lane
from .utils import format_hms, log_time


def _require_sam3_weights(path: Path = SAM3_MODEL) -> Path:
    resolved = path if path.is_absolute() else Path.cwd() / path
    if resolved.is_file():
        return resolved
    raise FileNotFoundError(
        f"SAM 3 weights not found at {resolved}.\n"
        "Ultralytics does not auto-download them. Request access and download "
        "sam3.pt from Hugging Face (Meta SAM 3), place it in the project root, "
        "then re-run --sam3-after-cut.\n"
        "Docs: https://docs.ultralytics.com/models/sam-3/"
    )


def _boxes_from_sam3_result(result) -> list[tuple[tuple[int, int, int, int], float]]:
    """Normalize one Ultralytics result into ((x1,y1,x2,y2), conf) boxes."""
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return boxes
    xyxy = result.boxes.xyxy
    conf = result.boxes.conf
    if xyxy is None:
        return boxes
    xyxy = xyxy.cpu().tolist()
    conf_list = (
        conf.cpu().tolist()
        if conf is not None
        else [1.0] * len(xyxy)
    )
    for (x1, y1, x2, y2), score in zip(xyxy, conf_list):
        boxes.append(((int(x1), int(y1), int(x2), int(y2)), float(score)))
    return boxes


def preview_sam3_after_first_cut(
    video_path: Path,
    lane: int,
    preview_seconds: float | None = None,
    after_cut_seconds: float = SAM3_AFTER_CUT_SECONDS,
) -> Path | None:
    """
    Run SAM 3 text prompts after the first cut and keep the requested lane.

    ``preview_seconds`` is the absolute end time in the source video
    (default: first_cut + after_cut_seconds). SAM 3 only runs between the
    first cut and that end time.
    """
    if not 1 <= lane <= LANE_COUNT:
        raise ValueError(f"lane must be an integer in 1..{LANE_COUNT}")

    weights = _require_sam3_weights()
    print(
        f"Finding the first camera cut "
        f"(lane {lane}: "
        f"{'closest/bottom' if lane == LANE_COUNT else 'furthest/top' if lane == 1 else 'mid'} "
        f"on a side camera)..."
    )
    search_window = preview_seconds if preview_seconds is not None else 60.0
    cuts = detect_cuts(video_path, max_seconds=search_window)
    if not cuts:
        print("No hard cut found; SAM 3 lane preview skipped.")
        return None
    first_cut = cuts[0]
    # Cap the SAM 3 window: run only a few seconds after the cut by default
    # (CPU-heavy), but never past the absolute --preview-seconds end time.
    end_time = first_cut.time_seconds + after_cut_seconds
    if preview_seconds is not None:
        end_time = min(end_time, preview_seconds)
    end_time = max(end_time, first_cut.time_seconds + 1.0 / 30.0)
    print(
        f"First cut: {first_cut.time_seconds:.3f}s (frame {first_cut.frame_index}). "
        f"SAM 3 window: {first_cut.time_seconds:.3f}s -> {end_time:.3f}s."
    )

    print(
        f"Loading SAM 3 ({weights.name}) with text prompts "
        f"{SAM3_TEXT_PROMPTS}..."
    )
    from ultralytics.models.sam import SAM3SemanticPredictor

    predictor = SAM3SemanticPredictor(
        overrides={
            "conf": SAM3_CONFIDENCE,
            "task": "segment",
            "mode": "predict",
            "imgsz": SAM3_IMAGE_SIZE,
            "model": str(weights),
            "verbose": False,
            "save": False,
        }
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = min(
        int(round(end_time * fps)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 2**31 - 1,
    )

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    out_dir = OUTPUT_FOLDER / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{video_path.stem}_sam3_lane{lane}_after_cut.mp4"
    )

    band_y0, band_y1 = lane_band_y_range(
        lane,
        height,
        LANE_COUNT,
        near_is_bottom=SAM3_NEAR_LANE_IS_BOTTOM,
        pool_y_top_frac=SAM3_POOL_Y_TOP_FRAC,
        pool_y_bottom_frac=SAM3_POOL_Y_BOTTOM_FRAC,
    )

    last_boxes: list[tuple[tuple[int, int, int, int], float]] = []
    chosen: tuple[tuple[int, int, int, int], float] | None = None
    hits = 0
    inference_frames = 0
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw_sam3_lane.mp4"
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create temporary SAM 3 preview video")

        frame_index = 0
        while frame_index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            absolute_time = frame_index / fps

            if frame_index >= first_cut.frame_index:
                relative = frame_index - first_cut.frame_index
                if relative % SAM3_EVERY_N_FRAMES == 0:
                    predictor.set_image(frame)
                    results = predictor(text=list(SAM3_TEXT_PROMPTS))
                    result = results[0] if isinstance(results, list) else results
                    last_boxes = _boxes_from_sam3_result(result)
                    inference_frames += 1
                    picked = pick_best_for_lane(
                        last_boxes,
                        lane,
                        height,
                        LANE_COUNT,
                        near_is_bottom=SAM3_NEAR_LANE_IS_BOTTOM,
                        pool_y_top_frac=SAM3_POOL_Y_TOP_FRAC,
                        pool_y_bottom_frac=SAM3_POOL_Y_BOTTOM_FRAC,
                    )
                    if picked is not None:
                        chosen = picked
                        hits += 1

                # Draw the requested lane band so the geometry is auditable.
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

                for box, confidence in last_boxes:
                    x1, y1, x2, y2 = box
                    center_y = 0.5 * (y1 + y2)
                    assigned = lane_from_center_y(
                        center_y,
                        height,
                        LANE_COUNT,
                        near_is_bottom=SAM3_NEAR_LANE_IS_BOTTOM,
                        pool_y_top_frac=SAM3_POOL_Y_TOP_FRAC,
                        pool_y_bottom_frac=SAM3_POOL_Y_BOTTOM_FRAC,
                    )
                    color = (80, 80, 80)  # outside pool / other lanes
                    if assigned == lane:
                        color = (255, 0, 255)
                    elif assigned is not None:
                        color = (180, 180, 180)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = (
                        f"L{assigned} {confidence:.2f}"
                        if assigned is not None
                        else f"? {confidence:.2f}"
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 22)),
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

                status = (
                    f"SAM3 swimmer | lane {lane} | t={absolute_time:.2f}s | "
                    f"raw={len(last_boxes)} | lane_hits={hits}"
                )
            else:
                status = (
                    f"Waiting for first cut at {first_cut.time_seconds:.3f}s | "
                    f"t={absolute_time:.2f}s | target lane {lane}"
                )

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
                remaining = elapsed / frame_index * (end_frame - frame_index)
                print(
                    f"  {frame_index}/{end_frame} frames, "
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
        f"SAM 3 lane preview ({frame_index} frames; "
        f"{inference_frames} inference frames)",
        start,
    )
    if chosen is None:
        print(
            f"No SAM 3 detection fell into lane {lane} "
            f"(pool y {SAM3_POOL_Y_TOP_FRAC:.0%}-{SAM3_POOL_Y_BOTTOM_FRAC:.0%})."
        )
    else:
        print(
            f"Lane {lane} selected at least once "
            f"({hits} inference frames with a lane hit)."
        )
    print(f"Saved SAM 3 lane preview: {out_path}")
    return out_path
