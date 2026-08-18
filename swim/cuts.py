"""Hard camera-cut detection and audit artifacts.

This detector deliberately uses whole-frame signals. A swimmer, splash, or
official can move a lot inside a small region; a camera edit changes most of
the image at once. For each pair of consecutive frames we calculate:

1. ``hist_distance`` — Bhattacharyya distance between HSV histograms.
2. ``pixel_difference`` — mean absolute grayscale difference.
3. ``edge_change`` — fraction of Canny edge pixels that disagree.

The first two signals must BOTH clear a floor. Their weighted score must also
be a local peak. This is aimed at hard broadcast cuts; slow dissolves are not
yet treated as cuts.
"""

import csv
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from .config import (
    CUT_DETECT_HEIGHT,
    CUT_DETECT_WIDTH,
    CUT_HIST_MIN,
    CUT_MIN_GAP_SECONDS,
    CUT_PEAK_RADIUS_FRAMES,
    CUT_PIXEL_MIN,
    CUT_SCORE_MIN,
)


class CutMetric(NamedTuple):
    """Change measurements for the transition into ``frame_index``."""

    frame_index: int
    time_seconds: float
    hist_distance: float
    pixel_difference: float
    edge_change: float
    score: float


class CameraCut(NamedTuple):
    """A detected first frame of a new camera shot."""

    frame_index: int
    time_seconds: float
    score: float
    hist_distance: float
    pixel_difference: float
    edge_change: float


def _small_frame_signals(
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grayscale, normalized HSV histogram, and edges for one frame."""
    small = cv2.resize(
        frame,
        (CUT_DETECT_WIDTH, CUT_DETECT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # H/S/V jointly capture colour, colourfulness, AND brightness. Including V
    # matters for cuts between two low-colour shots with different exposure.
    histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [16, 16, 16],
        [0, 180, 0, 256, 0, 256],
    )
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    edges = cv2.Canny(gray, 80, 160)
    return gray, histogram, edges


def _edge_disagreement(previous: np.ndarray, current: np.ndarray) -> float:
    """Return changed edge pixels divided by all edge pixels in either frame."""
    union = np.count_nonzero(cv2.bitwise_or(previous, current))
    if union == 0:
        return 0.0
    changed = np.count_nonzero(cv2.bitwise_xor(previous, current))
    return float(changed / union)


def measure_frame_changes(
    video_path: Path,
    max_seconds: float | None = None,
) -> tuple[list[CutMetric], float]:
    """Measure consecutive-frame transitions, optionally only to a time limit."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for cut detection: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    max_frames = (
        max(1, int(round(max_seconds * fps)))
        if max_seconds is not None
        else None
    )
    metrics: list[CutMetric] = []
    previous: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    frame_index = 0

    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        current = _small_frame_signals(frame)
        if previous is not None:
            prev_gray, prev_histogram, prev_edges = previous
            gray, histogram, edges = current
            hist_distance = float(
                cv2.compareHist(
                    prev_histogram,
                    histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
            pixel_difference = float(
                np.mean(cv2.absdiff(prev_gray, gray)) / 255.0
            )
            edge_change = _edge_disagreement(prev_edges, edges)

            # Colour/layout is the strongest clue. Pixel difference catches
            # same-colour angle changes; edges are only a supporting vote.
            score = (
                0.55 * hist_distance
                + 0.35 * pixel_difference
                + 0.10 * edge_change
            )
            metrics.append(
                CutMetric(
                    frame_index,
                    frame_index / fps,
                    hist_distance,
                    pixel_difference,
                    edge_change,
                    score,
                )
            )
        previous = current
        frame_index += 1

    capture.release()
    if frame_index == 0:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return metrics, fps


def select_hard_cuts(metrics: list[CutMetric], fps: float) -> list[CameraCut]:
    """Apply thresholds, local-peak filtering, and temporal de-duplication."""
    candidates: list[CutMetric] = []
    radius = CUT_PEAK_RADIUS_FRAMES

    for index, metric in enumerate(metrics):
        # Requiring both global signals is the main splash/motion rejection.
        if (
            metric.hist_distance < CUT_HIST_MIN
            or metric.pixel_difference < CUT_PIXEL_MIN
            or metric.score < CUT_SCORE_MIN
        ):
            continue

        lo = max(0, index - radius)
        hi = min(len(metrics), index + radius + 1)
        if metric.score < max(item.score for item in metrics[lo:hi]):
            continue
        candidates.append(metric)

    min_gap_frames = max(1, int(round(CUT_MIN_GAP_SECONDS * fps)))
    accepted: list[CutMetric] = []
    for candidate in candidates:
        if accepted and candidate.frame_index - accepted[-1].frame_index < min_gap_frames:
            # If two peaks describe one edit, retain the stronger transition.
            if candidate.score > accepted[-1].score:
                accepted[-1] = candidate
            continue
        accepted.append(candidate)

    return [
        CameraCut(
            item.frame_index,
            item.time_seconds,
            item.score,
            item.hist_distance,
            item.pixel_difference,
            item.edge_change,
        )
        for item in accepted
    ]


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = capture.read()
    return frame if ok else None


def save_cut_audit(
    video_path: Path,
    output_dir: Path,
    metrics: list[CutMetric],
    cuts: list[CameraCut],
) -> None:
    """Save all scores as CSV and before/after JPEGs for detected cuts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "frame_change_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame",
                "time_seconds",
                "hist_distance",
                "pixel_difference",
                "edge_change",
                "score",
                "is_cut",
            ]
        )
        cut_frames = {cut.frame_index for cut in cuts}
        for metric in metrics:
            writer.writerow(
                [
                    metric.frame_index,
                    f"{metric.time_seconds:.6f}",
                    f"{metric.hist_distance:.6f}",
                    f"{metric.pixel_difference:.6f}",
                    f"{metric.edge_change:.6f}",
                    f"{metric.score:.6f}",
                    int(metric.frame_index in cut_frames),
                ]
            )

    capture = cv2.VideoCapture(str(video_path))
    for number, cut in enumerate(cuts, start=1):
        before = _read_frame(capture, cut.frame_index - 1)
        after = _read_frame(capture, cut.frame_index)
        if before is None or after is None:
            continue
        height = min(before.shape[0], after.shape[0])
        before = before[:height]
        after = after[:height]
        preview = np.hstack((before, after))
        label = (
            f"CUT {cut.time_seconds:.3f}s | score {cut.score:.3f} | "
            f"hist {cut.hist_distance:.3f} | pixels {cut.pixel_difference:.3f}"
        )
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            preview,
            label,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        filename = f"cut_{number:03d}_t{cut.time_seconds:09.3f}s.jpg"
        cv2.imwrite(str(output_dir / filename), preview)
    capture.release()


def detect_cuts(
    video_path: Path,
    output_dir: Path | None = None,
    max_seconds: float | None = None,
) -> list[CameraCut]:
    """Detect hard cuts, optionally saving an auditable CSV and previews."""
    metrics, fps = measure_frame_changes(video_path, max_seconds=max_seconds)
    cuts = select_hard_cuts(metrics, fps)
    if output_dir is not None:
        save_cut_audit(video_path, output_dir, metrics, cuts)
    return cuts
