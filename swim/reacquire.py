"""Re-acquire one lane after a camera cut: SAM 3 until lock -> SAM 2 both ways.

SAM 3 scans the camera shot (this cut to the next) with text='swimmer'. As soon
as a lane-matched box recurs with high enough confidence, that frame seeds
SAM 2. SAM 2 then tracks backward to the cut and forward to the next cut, so a
late lock (e.g. 15s in) still covers the full shot.

If SAM 2 later loses the mask, SAM 3 (and optional YOLO) re-seeds. The empty
gap between loss and the new seed is filled by tracking SAM 2 backward from
the new seed to the loss time, so those seconds are not left blank.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from .config import (
    LANE_COUNT,
    MAX_CENTER_JUMP_PX,
    OUTPUT_FOLDER,
    REACQUIRE_LOST_SECONDS,
    REACQUIRE_MATCH_CENTER_FRAME_FRAC,
    REACQUIRE_MATCH_IOU,
    REACQUIRE_MAX_RESEEDS,
    REACQUIRE_MIN_CONFIDENCE,
    REACQUIRE_POSITION_BASE_FRAME_FRAC,
    REACQUIRE_POSITION_GROWTH_FRAME_FRAC_PER_SECOND,
    REACQUIRE_SEED_EVERY_SECONDS,
    REACQUIRE_STABLE_HITS,
    REACQUIRE_VERIFY_CENTER_FRAME_FRAC,
    REACQUIRE_VERIFY_EVERY_SECONDS,
    REACQUIRE_VERIFY_MIN_CONFIDENCE,
    REACQUIRE_VERIFY_MISSES,
    REACQUIRE_YOLO_OVERLAP_IOU,
    SAM3_CONFIDENCE,
    SAM3_IMAGE_SIZE,
    SAM3_MODEL,
    SAM3_TEXT_PROMPTS,
    SAM_TRACK_MODEL,
)
from .cuts import CameraCut, detect_cuts
from .ropes import (
    RopeLine,
    RopeTemporalFilter,
    detect_lane_ropes,
    draw_rope_overlay,
    lane_for_point,
    lane_polygon,
)
from .boxtrack import BoxTrack, no_pose_flag
from .sam3_preview import _boxes_from_sam3_result, _require_sam3_weights
from .tracking import keep_one_person_mask
from .utils import format_hms, log_time


class ReacquireSeed(NamedTuple):
    """Stable SAM 3 box selected to initialize SAM 2."""

    frame_index: int
    time_seconds: float
    box: tuple[int, int, int, int]
    hits: int
    mean_confidence: float
    ropes: tuple[RopeLine, ...]


class SamClipResult(NamedTuple):
    """Result of one SAM 2 pass, including the first sustained mask outage."""

    written: int
    fps: float
    lost_time: float | None
    lost_xy: tuple[float, float] | None
    lost_velocity: tuple[float, float]
    lost_ropes: tuple[RopeLine, ...]
    lost_reason: str | None


class LaneVerification(NamedTuple):
    """Highest-confidence SAM 3 swimmer box in the requested lane."""

    time_seconds: float
    best_box: tuple[int, int, int, int] | None


class _CandidateTrack:
    """Small SAM 3 tracklet used only to reject one-frame detections."""

    def __init__(
        self,
        box: tuple[int, int, int, int],
        confidence: float,
        frame_index: int,
        ropes: list[RopeLine],
        yolo_supported: bool = False,
    ) -> None:
        self.last_box = box
        self.last_frame = frame_index
        self.last_ropes = tuple(ropes)
        self.hits = 1
        self.confidence_sum = confidence
        self.best_confidence = confidence
        self.best_box = box
        self.best_frame = frame_index
        self.best_ropes = tuple(ropes)
        self.yolo_hits = 1 if yolo_supported else 0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.hits

    def update(
        self,
        box: tuple[int, int, int, int],
        confidence: float,
        frame_index: int,
        ropes: list[RopeLine],
        yolo_supported: bool = False,
    ) -> None:
        self.last_box = box
        self.last_frame = frame_index
        self.last_ropes = tuple(ropes)
        self.hits += 1
        self.confidence_sum += confidence
        if yolo_supported:
            self.yolo_hits += 1
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_box = box
            self.best_frame = frame_index
            self.best_ropes = tuple(ropes)


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _box_center_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Euclidean distance between xyxy box centres."""
    first_x = 0.5 * (first[0] + first[2])
    first_y = 0.5 * (first[1] + first[3])
    second_x = 0.5 * (second[0] + second[2])
    second_y = 0.5 * (second[1] + second[3])
    return float(np.hypot(first_x - second_x, first_y - second_y))


def _optional_swimmer_yolo():
    """Load the specialized swimmer YOLO if weights exist; otherwise None."""
    try:
        from .detect import load_swimmer_yolov5

        model = load_swimmer_yolov5()
        print("  YOLO swimmer veto enabled.")
        return model
    except FileNotFoundError:
        print("  YOLO swimmer model unavailable; SAM 3-only seeding.")
        return None
    except Exception as exc:
        print(f"  YOLO swimmer model failed to load; SAM 3-only seeding. ({exc})")
        return None


def _boxes_in_lane(
    detections: list[tuple[tuple[int, int, int, int], float]],
    ropes: list[RopeLine],
    lane: int,
    closest_lane: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Keep boxes whose centre falls in the requested physical lane."""
    in_lane: list[tuple[tuple[int, int, int, int], float]] = []
    for box, confidence in detections:
        center_x = 0.5 * (box[0] + box[2])
        center_y = 0.5 * (box[1] + box[3])
        assigned = lane_for_point(
            ropes,
            center_x,
            center_y,
            closest_lane=closest_lane,
        )
        if assigned == lane:
            in_lane.append((box, confidence))
    return in_lane


def _yolo_lane_boxes(
    yolo_model,
    frame: np.ndarray,
    ropes: list[RopeLine],
    lane: int,
    closest_lane: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Run swimmer YOLO and keep boxes that sit in ``lane``."""
    if yolo_model is None:
        return []
    from .detect import _yolov5_boxes

    raw: list[tuple[tuple[int, int, int, int], float]] = []
    for xyxy, confidence, _label in _yolov5_boxes(yolo_model, frame):
        box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
        raw.append((box, float(confidence)))
    return _boxes_in_lane(raw, ropes, lane, closest_lane)


def _position_near_last_xy(
    box: tuple[int, int, int, int],
    frame_width: int,
    expected_xy: tuple[float, float] | None,
    expected_time: float | None,
    now: float,
) -> bool:
    """True if the box is close enough to the last known SAM 2 position.

    Velocity is ignored on purpose: after SAM 2 latches onto wake, the speed
    estimate points at foam and would reject the real swimmer.
    """
    if expected_xy is None or expected_time is None:
        return True
    elapsed = max(0.0, now - expected_time)
    allowance = frame_width * (
        REACQUIRE_POSITION_BASE_FRAME_FRAC
        + REACQUIRE_POSITION_GROWTH_FRAME_FRAC_PER_SECOND * elapsed
    )
    center_x = 0.5 * (box[0] + box[2])
    return abs(center_x - expected_xy[0]) <= allowance


def _merge_sam3_with_yolo(
    sam3: list[tuple[tuple[int, int, int, int], float]],
    yolo: list[tuple[tuple[int, int, int, int], float]],
) -> list[tuple[tuple[int, int, int, int], float, bool]]:
    """Veto foam SAM 3 boxes when YOLO sees a person in the lane.

    If YOLO is empty, SAM 3 is used alone. If YOLO has a detection but no
    SAM 3 box overlaps it, YOLO itself becomes the seed source.
    """
    if not yolo:
        return [(box, confidence, False) for box, confidence in sam3]
    supported: list[tuple[tuple[int, int, int, int], float, bool]] = []
    for box, confidence in sam3:
        if any(
            _box_iou(box, yolo_box) >= REACQUIRE_YOLO_OVERLAP_IOU
            for yolo_box, _yolo_conf in yolo
        ):
            supported.append((box, confidence, True))
    if supported:
        return supported
    return [(box, confidence, True) for box, confidence in yolo]


def _same_identity(
    sam2_box: tuple[int, int, int, int],
    sam3_box: tuple[int, int, int, int],
    frame_width: int,
) -> bool:
    """True if SAM 2 and the best SAM 3 lane box are the same body."""
    center_distance = _box_center_distance(sam2_box, sam3_box)
    center_limit = max(
        REACQUIRE_VERIFY_CENTER_FRAME_FRAC * frame_width,
        1.5
        * max(
            sam2_box[2] - sam2_box[0],
            sam3_box[2] - sam3_box[0],
        ),
    )
    return (
        _box_iou(sam2_box, sam3_box) >= 0.05 or center_distance <= center_limit
    )


def _usable_ropes(
    detected: list[RopeLine] | None,
    rope_filter: RopeTemporalFilter,
    frame_width: int,
    frame: np.ndarray,
    last_good: list[RopeLine],
) -> list[RopeLine]:
    """Keep the last valid lane ladder when the current frame has no ropes."""
    filtered = rope_filter.update(detected or [], frame_width, frame=frame)
    if filtered is not None and len(filtered) == LANE_COUNT + 1:
        last_good[:] = list(filtered)
        return last_good
    return last_good


def find_stable_lane_seed(
    video_path: Path,
    first_cut: CameraCut,
    lane: int,
    closest_lane: int,
    scan_end_frame: int,
    initial_ropes: tuple[RopeLine, ...] = (),
    expected_xy: tuple[float, float] | None = None,
    expected_time: float | None = None,
    allow_weak_fallback: bool = True,
    yolo_model=None,
    prior_confident: bool = False,
) -> ReacquireSeed | None:
    """Scan the shot with SAM 3 until a recurring, confident box appears in ``lane``.

    Samples about every ``REACQUIRE_SEED_EVERY_SECONDS``. Stops early once a
    tracklet has enough hits and its best confidence is at least
    ``REACQUIRE_MIN_CONFIDENCE``. YOLO, when available, vetoes foam boxes.

    prior_confident: cross-shot handoff — the SAME lane was confidently
    tracked right before this cut (proven by box CSVs on disk), so the
    prior "an active racer is in this lane" is already strong and ONE
    high-bar hit may seed (conf >= 0.75, or conf >= 0.60 with YOLO
    agreement) instead of waiting a full stride for the second one.
    """
    weights = _require_sam3_weights()
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
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    scan_end = scan_end_frame
    if total_frames > 0:
        scan_end = min(scan_end, total_frames)
    sample_stride = max(1, int(round(REACQUIRE_SEED_EVERY_SECONDS * fps)))
    # NOTE: blanket denser scanning on short shots was TESTED (cut 7) and
    # reverted — 12 extra SAM 3 scans found nothing, because a gliding
    # breaststroker is invisible to the text detector. The bottleneck is
    # visibility, not cadence; the real fix is a cross-shot position
    # handoff (seed the next shot from the previous shot's last position).
    next_scan_at = first_cut.frame_index
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_cut.frame_index)

    tracklets: list[_CandidateTrack] = []
    rope_filter = RopeTemporalFilter()
    rope_filter_initialized = False
    last_good_ropes: list[RopeLine] = (
        list(initial_ropes) if len(initial_ropes) == LANE_COUNT + 1 else []
    )
    frame_index = first_cut.frame_index
    max_missing = sample_stride * 2

    def _ready(tracklet: _CandidateTrack) -> bool:
        if prior_confident and tracklet.hits >= 1:
            if tracklet.best_confidence >= 0.75:
                return True
            if tracklet.best_confidence >= 0.60 and tracklet.yolo_hits >= 1:
                return True
        if tracklet.hits < REACQUIRE_STABLE_HITS:
            return False
        if tracklet.best_confidence >= REACQUIRE_MIN_CONFIDENCE:
            return True
        # Two SAM 3 hits plus at least one overlapping YOLO box is enough
        # even when SAM 3 itself stays a bit under the confidence floor.
        return tracklet.yolo_hits >= 1

    while frame_index < scan_end:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index < next_scan_at:
            frame_index += 1
            continue
        # After the first candidate appears, densify: the 2nd confirming hit
        # should come as soon as possible, not a full stride later.
        stride_now = sample_stride if not tracklets else max(1, sample_stride // 2)
        next_scan_at = frame_index + stride_now

        detected_ropes = detect_lane_ropes(frame, lane_count=LANE_COUNT)
        if not rope_filter_initialized:
            if last_good_ropes:
                rope_filter.update(list(last_good_ropes), frame.shape[1])
            rope_filter_initialized = True
        ropes = _usable_ropes(
            detected_ropes,
            rope_filter,
            frame.shape[1],
            frame,
            last_good_ropes,
        )
        if len(ropes) != LANE_COUNT + 1:
            frame_index += 1
            continue

        now = frame_index / fps
        print(
            f"  SAM 3 seed scan t={now:.2f}s "
            f"(frame {frame_index})...",
            flush=True,
        )
        predictor.set_image(frame)
        results = predictor(text=list(SAM3_TEXT_PROMPTS))
        result = results[0] if isinstance(results, list) else results
        sam3_lane = _boxes_in_lane(
            _boxes_from_sam3_result(result),
            ropes,
            lane,
            closest_lane,
        )
        sam3_lane = [
            (box, confidence)
            for box, confidence in sam3_lane
            if _position_near_last_xy(
                box,
                frame.shape[1],
                expected_xy,
                expected_time,
                now,
            )
        ]
        yolo_lane = _yolo_lane_boxes(
            yolo_model,
            frame,
            ropes,
            lane,
            closest_lane,
        )
        yolo_lane = [
            (box, confidence)
            for box, confidence in yolo_lane
            if _position_near_last_xy(
                box,
                frame.shape[1],
                expected_xy,
                expected_time,
                now,
            )
        ]
        lane_candidates = _merge_sam3_with_yolo(sam3_lane, yolo_lane)
        if yolo_lane:
            print(
                f"    lane {lane}: SAM3={len(sam3_lane)} YOLO={len(yolo_lane)} "
                f"kept={len(lane_candidates)}",
                flush=True,
            )

        # Associate by overlap OR centre motion. IoU alone fragments a real
        # swimmer into one-hit tracklets when they move farther than their
        # narrow SAM 3 box between 0.5s samples.
        used_tracklets: set[int] = set()
        for box, confidence, yolo_supported in sorted(
            lane_candidates, key=lambda item: item[1], reverse=True
        ):
            best_index = None
            best_score = float("-inf")
            for index, tracklet in enumerate(tracklets):
                if index in used_tracklets:
                    continue
                if tracklet.last_frame == frame_index:
                    continue
                if frame_index - tracklet.last_frame > max_missing:
                    continue
                iou = _box_iou(box, tracklet.last_box)
                center_distance = _box_center_distance(box, tracklet.last_box)
                center_limit = max(
                    REACQUIRE_MATCH_CENTER_FRAME_FRAC * frame.shape[1],
                    1.5
                    * max(
                        box[2] - box[0],
                        box[3] - box[1],
                        tracklet.last_box[2] - tracklet.last_box[0],
                        tracklet.last_box[3] - tracklet.last_box[1],
                    ),
                )
                if iou < REACQUIRE_MATCH_IOU and center_distance > center_limit:
                    continue
                score = iou - center_distance / max(center_limit, 1.0)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None:
                tracklets.append(
                    _CandidateTrack(
                        box,
                        confidence,
                        frame_index,
                        ropes,
                        yolo_supported=yolo_supported,
                    )
                )
                used_tracklets.add(len(tracklets) - 1)
                ready = tracklets[-1]
            else:
                tracklets[best_index].update(
                    box,
                    confidence,
                    frame_index,
                    ropes,
                    yolo_supported=yolo_supported,
                )
                used_tracklets.add(best_index)
                ready = tracklets[best_index]

            # Lock as soon as one tracklet is recurring and confident enough.
            if _ready(ready):
                capture.release()
                return ReacquireSeed(
                    # Seed from the latest observation, not the highest-conf
                    # historical box. In test2 the early high-conf box was
                    # wake behind the swimmer; the recurring latest box had
                    # expanded onto the actual body.
                    frame_index=ready.last_frame,
                    time_seconds=ready.last_frame / fps,
                    box=ready.last_box,
                    hits=ready.hits,
                    mean_confidence=ready.mean_confidence,
                    ropes=ready.last_ropes,
                )
        frame_index += 1

    capture.release()
    # Shot ended without an early lock; accept the best recurring tracklet
    # even if it stayed under the confidence floor.
    if not allow_weak_fallback:
        return None
    stable = [
        tracklet
        for tracklet in tracklets
        if tracklet.hits >= REACQUIRE_STABLE_HITS
    ]
    if not stable:
        return None
    best = max(
        stable,
        key=lambda item: (item.yolo_hits, item.best_confidence, item.hits),
    )
    return ReacquireSeed(
        frame_index=best.last_frame,
        time_seconds=best.last_frame / fps,
        box=best.last_box,
        hits=best.hits,
        mean_confidence=best.mean_confidence,
        ropes=best.last_ropes,
    )


def _scan_lane_verifications(
    video_path: Path,
    start_time: float,
    end_time: float,
    lane: int,
    closest_lane: int,
    initial_ropes: tuple[RopeLine, ...],
) -> list[LaneVerification]:
    """Run sparse SAM 3 checkpoints used to catch a non-empty wake mask."""
    if end_time - start_time < REACQUIRE_VERIFY_EVERY_SECONDS:
        return []
    weights = _require_sam3_weights()
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
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    rope_filter = RopeTemporalFilter()
    initialized = False
    last_good_ropes: list[RopeLine] = (
        list(initial_ropes) if len(initial_ropes) == LANE_COUNT + 1 else []
    )
    verifications: list[LaneVerification] = []
    check_time = start_time + REACQUIRE_VERIFY_EVERY_SECONDS
    while check_time < end_time - 1.0 / fps:
        frame_index = int(round(check_time * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        if not initialized:
            if last_good_ropes:
                rope_filter.update(list(last_good_ropes), frame.shape[1])
            initialized = True
        detected = detect_lane_ropes(frame, lane_count=LANE_COUNT)
        ropes = _usable_ropes(
            detected,
            rope_filter,
            frame.shape[1],
            frame,
            last_good_ropes,
        )
        best_box: tuple[int, int, int, int] | None = None
        if len(ropes) == LANE_COUNT + 1:
            print(f"  SAM 3 identity check t={check_time:.2f}s...", flush=True)
            predictor.set_image(frame)
            results = predictor(text=list(SAM3_TEXT_PROMPTS))
            result = results[0] if isinstance(results, list) else results
            lane_boxes = [
                (box, confidence)
                for box, confidence in _boxes_from_sam3_result(result)
                if confidence >= REACQUIRE_VERIFY_MIN_CONFIDENCE
            ]
            lane_boxes = _boxes_in_lane(lane_boxes, ropes, lane, closest_lane)
            if lane_boxes:
                best_box = max(lane_boxes, key=lambda item: item[1])[0]
        verifications.append(LaneVerification(check_time, best_box))
        check_time += REACQUIRE_VERIFY_EVERY_SECONDS
    capture.release()
    return verifications


def _save_seed_audit(
    video_path: Path,
    output_path: Path,
    seed: ReacquireSeed,
    lane: int,
    closest_lane: int,
) -> None:
    """Save the exact SAM 3 frame and box that will be passed to SAM 2."""
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, seed.frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return
    frame = draw_rope_overlay(
        frame,
        list(seed.ropes),
        target_lane=lane,
        closest_lane=closest_lane,
    )
    x1, by1, x2, by2 = seed.box
    cv2.rectangle(frame, (x1, by1), (x2, by2), (0, 255, 255), 4)
    cv2.putText(
        frame,
        (
            f"SAM3 -> SAM2 | lane {lane} | t={seed.time_seconds:.3f}s | "
            f"hits={seed.hits} mean={seed.mean_confidence:.2f}"
        ),
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), frame)


def _annotate_sam_clip(
    clip_path: Path,
    raw_path: Path,
    seed: ReacquireSeed,
    lane: int,
    closest_lane: int,
    clip_start_time: float,
    time_direction: int,
    expected_frames: int,
    skip_first_frame: bool = False,
    rope_filter: RopeTemporalFilter | None = None,
    detect_loss: bool = False,
    verifications: tuple[LaneVerification, ...] = (),
    csv_dir: Path | None = None,
    single_lane: bool = False,
    label: str | None = None,
) -> SamClipResult:
    """Track one clip with SAM 2 from the seed box and write an annotated video.

    ``time_direction`` is -1 for a time-reversed clip, so the on-screen clock
    still counts in real time while SAM walks backwards towards the cut.
    """
    from ultralytics.models.sam import SAM2VideoPredictor

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
    results = predictor(
        source=str(clip_path),
        bboxes=list(seed.box),
        stream=True,
    )

    capture = cv2.VideoCapture(str(clip_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()

    writer = None
    seen = 0
    written = 0
    first_area = None
    box_track = BoxTrack(fps=fps, time_direction=time_direction)
    box_rows: list[str] = []
    pred_x = 0.5 * (seed.box[0] + seed.box[2])
    pred_y = 0.5 * (seed.box[1] + seed.box[3])
    velocity_x = 0.0
    velocity_y = 0.0
    max_jump = max(
        MAX_CENTER_JUMP_PX,
        0.55 * max(seed.box[2] - seed.box[0], seed.box[3] - seed.box[1]),
    )
    rope_filter = rope_filter or RopeTemporalFilter()
    ropes_now: list[RopeLine] | None = (
        list(seed.ropes) if seed.ropes else None
    )
    rope_filter_initialized = False
    has_tracked = False
    lost_run = 0
    lost_run_start: float | None = None
    sustained_lost_time: float | None = None
    lost_xy: tuple[float, float] | None = None
    lost_velocity = (0.0, 0.0)
    lost_ropes: tuple[RopeLine, ...] = ()
    lost_reason: str | None = None
    last_good_ropes: tuple[RopeLine, ...] = tuple(seed.ropes)
    verification_index = 0
    verification_misses = 0
    first_verification_miss: float | None = None
    start = time.perf_counter()

    for result in results:
        source_frame = result.orig_img
        frame = source_frame.copy()
        height, width = frame.shape[:2]
        if not rope_filter_initialized and not single_lane:
            ropes_now = rope_filter.update(list(seed.ropes), frame_width=width)
            rope_filter_initialized = True
        if writer is None:
            writer = cv2.VideoWriter(
                str(raw_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        if seen % 3 == 0 and not single_lane:
            # Keep detecting; the temporal filter follows real camera motion,
            # suppresses isolated bad fits, and bridges only short outages.
            detected_ropes = detect_lane_ropes(source_frame, lane_count=LANE_COUNT)
            ropes_now = rope_filter.update(
                (
                    detected_ropes
                    if len(detected_ropes) == LANE_COUNT + 1
                    else []
                ),
                frame_width=width,
                frame=source_frame,
            )
            if ropes_now is not None and len(ropes_now) == LANE_COUNT + 1:
                last_good_ropes = tuple(ropes_now)
            elif last_good_ropes:
                ropes_now = list(last_good_ropes)

        tracked = False
        current_box: tuple[int, int, int, int] | None = None
        if result.masks is not None and len(result.masks) > 0:
            mask = result.masks.data[0].cpu().numpy().astype(np.float32)
            mask = cv2.resize(mask, (width, height))
            mask = keep_one_person_mask(
                mask,
                (pred_x, pred_y),
                first_area,
                max_jump,
            )
            if ropes_now is not None and len(ropes_now) == LANE_COUNT + 1:
                # Hard spatial guard: even a valid-looking SAM blob may not
                # consume pixels from a neighbouring physical lane.
                polygon = lane_polygon(
                    ropes_now,
                    lane,
                    width,
                    closest_lane,
                )
                lane_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(lane_mask, [polygon], 1)
                mask[lane_mask == 0] = 0.0
            area = float(np.count_nonzero(mask > 0.5))
            if area > 0:
                ys, xs = np.where(mask > 0.5)
                center_x = float(xs.mean())
                center_y = float(ys.mean())
                assigned_lane = (
                    lane_for_point(
                        ropes_now,
                        center_x,
                        center_y,
                        closest_lane=closest_lane,
                    )
                    if ropes_now is not None
                    else lane
                )
                # Never allow a SAM 2 identity switch into another physical
                # lane. Losing the mask is safer: it invokes lane-filtered
                # SAM 3 re-acquisition with clean predictor memory.
                if assigned_lane == lane:
                    if first_area is None:
                        first_area = area
                    frame_dx = center_x - pred_x
                    frame_dy = center_y - pred_y
                    velocity_x = (
                        0.8 * velocity_x
                        + 0.2 * frame_dx * fps * time_direction
                    )
                    velocity_y = (
                        0.8 * velocity_y
                        + 0.2 * frame_dy * fps * time_direction
                    )
                    pred_x = center_x
                    pred_y = center_y
                    colored = frame.copy()
                    colored[mask > 0.5] = (0, 180, 255)
                    frame = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()), int(ys.max())
                    current_box = (x1, y1, x2, y2)
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 255), 3
                    )
                    tracked = True

        absolute_time = clip_start_time + time_direction * seen / fps
        while (
            verification_index < len(verifications)
            and absolute_time + 0.5 / fps
            >= verifications[verification_index].time_seconds
        ):
            verification = verifications[verification_index]
            verification_index += 1
            # Compare SAM 2 only against the strongest SAM 3 box in this lane.
            # Foam often appears as a second, weaker box; matching any overlap
            # would let a wake mask look legitimate. An empty SAM 2 mask at a
            # checkpoint where SAM 3 sees a swimmer is a miss.
            if verification.best_box is None:
                continue
            compatible = (
                current_box is not None
                and _same_identity(current_box, verification.best_box, width)
            )
            if compatible:
                verification_misses = 0
                first_verification_miss = None
            else:
                if verification_misses == 0:
                    first_verification_miss = verification.time_seconds
                verification_misses += 1
                if (
                    detect_loss
                    and sustained_lost_time is None
                    and verification_misses >= REACQUIRE_VERIFY_MISSES
                ):
                    sustained_lost_time = first_verification_miss
                    lost_xy = (pred_x, pred_y)
                    lost_velocity = (velocity_x, velocity_y)
                    lost_ropes = last_good_ropes
                    lost_reason = "SAM3 identity mismatch"
        if tracked:
            has_tracked = True
            lost_run = 0
            lost_run_start = None
        elif has_tracked:
            if lost_run == 0:
                lost_run_start = absolute_time
                lost_xy = (pred_x, pred_y)
                lost_velocity = (velocity_x, velocity_y)
            lost_run += 1
            lost_frames = max(1, int(round(REACQUIRE_LOST_SECONDS * fps)))
            if (
                detect_loss
                and sustained_lost_time is None
                and lost_run >= lost_frames
            ):
                sustained_lost_time = lost_run_start
                lost_ropes = last_good_ropes
                lost_reason = "empty SAM2 mask"
        seen += 1
        if skip_first_frame and seen == 1:
            continue

        smooth = box_track.update(
            current_box,
            ropes_now if ropes_now and len(ropes_now) == LANE_COUNT + 1 else None,
            lane,
            closest_lane,
            (width, height),
        )
        no_pose = False
        if smooth.box is not None:
            lane_mask_np: np.ndarray | None = None
            if ropes_now is not None and len(ropes_now) == LANE_COUNT + 1:
                polygon = lane_polygon(ropes_now, lane, width, closest_lane)
                lane_mask_np = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(lane_mask_np, [polygon], 1)
            no_pose = no_pose_flag(source_frame, smooth.box, lane_mask_np)
            color = (
                (140, 140, 140)
                if no_pose
                else (0, 255, 0) if smooth.state == "TRACKING" else (0, 165, 255)
            )
            bx1, by1, bx2, by2 = smooth.box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
            tag = "NO-POSE" if no_pose else smooth.state
            if smooth.length_m is not None:
                tag += f" {smooth.length_m:.1f}m"
            cv2.putText(
                frame,
                tag,
                (bx1, max(by1 - 8, 58)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        raw_txt = (
            ",".join(str(v) for v in current_box) if current_box else ",,,"
        )
        smooth_txt = (
            ",".join(str(v) for v in smooth.box) if smooth.box else ",,,"
        )
        box_rows.append(
            f"{absolute_time:.3f},{raw_txt},{smooth_txt},"
            f"{smooth.state},{int(no_pose)},"
            f"{'' if smooth.length_m is None else f'{smooth.length_m:.2f}'}"
        )

        if ropes_now is not None:
            frame = draw_rope_overlay(
                frame,
                ropes_now,
                target_lane=lane,
                closest_lane=closest_lane,
            )
        status = (
            f"SAM3 -> SAM2 | {'SINGLE-LANE (close-up)' if single_lane else f'lane {lane}'} | nearest={closest_lane} | "
            f"t={absolute_time:.2f}s | "
            f"{'IDENTITY DRIFT' if lost_reason == 'SAM3 identity mismatch' else ('TRACKING' if tracked else 'MASK LOST')}"
            f"{f' | laneQ={ropes_now[0].score:.2f}' if ropes_now else ' | no lane geometry'}"
        )
        cv2.rectangle(frame, (0, 0), (width, 44), (0, 0, 0), -1)
        cv2.putText(
            frame,
            status,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        written += 1
        if written % 30 == 0:
            elapsed = time.perf_counter() - start
            remaining = elapsed / written * max(expected_frames - written, 0)
            print(
                f"  SAM 2 {written}/{expected_frames} frames, "
                f"elapsed {format_hms(elapsed)}, ETA {format_hms(remaining)}"
            )
        if detect_loss and sustained_lost_time is not None:
            # Loss is confirmed. Stop instead of walking the rest of the shot
            # with an empty/foam mask; SAM 3 re-seed + backward fill covers
            # the gap.
            break

    if writer is not None:
        writer.release()
    if box_rows:
        csv_home = csv_dir if csv_dir is not None else raw_path.parent
        run_tag = (
            f"{label or f'lane{lane}'}_seed{seed.time_seconds:.2f}s"
            f"_dir{time_direction:+d}"
        )
        csv_path = csv_home / f"{run_tag}_{raw_path.stem}_boxes.csv"
        csv_path.write_text(
            "time_s,raw_x1,raw_y1,raw_x2,raw_y2,"
            "smooth_x1,smooth_y1,smooth_x2,smooth_y2,"
            "state,no_pose,length_m\n" + "\n".join(box_rows) + "\n"
        )
        print(f"  Box CSV: {csv_path}")
    return SamClipResult(
        written,
        fps,
        sustained_lost_time,
        lost_xy,
        lost_velocity,
        lost_ropes,
        lost_reason,
    )


def _extract_clip(
    video_path: Path,
    clip_path: Path,
    start_time: float,
    duration: float,
    reverse: bool = False,
) -> None:
    """Cut a clip, optionally time-reversed so it starts at the seed frame.

    ``-ss``/``-t`` are input options so the trim happens before ``reverse``;
    as output options the filter would reverse the whole file first.
    """
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(video_path),
        "-an",
    ]
    if reverse:
        command += ["-vf", "reverse"]
    command.append(str(clip_path))
    subprocess.run(command, check=True, capture_output=True)


def _ffmpeg_reverse(source: Path, dest: Path) -> None:
    """Flip a clip in time so a backward SAM 2 pass plays forward."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-vf", "reverse", str(dest)],
        check=True,
        capture_output=True,
    )


def _video_duration_seconds(video_path: Path) -> float:
    """Duration of a video in seconds."""
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    return frames / fps if fps > 0 else 0.0


def _close_up_shot(
    video_path: Path,
    first_cut: CameraCut,
    fps: float,
    shot_end: float,
) -> bool:
    """Close-up detector, measured on test2: a wide shot fits the ladder
    with 8-9 direct rope matches; a close-up either fails to fit at all or
    forces a ladder through only ~6 matches (and laneQ does NOT catch it).
    """
    from .ropes import _line_candidates, detect_lane_geometry

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    span = max(min(shot_end - first_cut.time_seconds, 2.5), 0.3)
    suspicious = 0
    candidate_counts: list[int] = []
    probes = 4
    for k in range(probes):
        t = first_cut.time_seconds + span * (k + 0.5) / probes
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = capture.read()
        if not ok:
            continue
        candidate_counts.append(len(_line_candidates(frame)))
        geo = detect_lane_geometry(frame, lane_count=LANE_COUNT)
        if geo is None or geo.direct_matches <= 6:
            suspicious += 1
    capture.release()
    # BOTH conditions must hold. A hard wide shot (splash/glare at a turn,
    # e.g. cut 7) also fails the ladder fit, but it still shows MANY
    # rope-like lines; a real close-up has few lanes in frame at all
    # (measured medians: close-up 10.5 vs wide 14).
    geo_bad = suspicious >= int(0.6 * probes) + 1
    few_ropes = (
        bool(candidate_counts)
        and float(np.median(candidate_counts)) <= 12.0
    )
    return geo_bad and few_ropes


def _find_dominant_seed(
    video_path: Path,
    first_cut: CameraCut,
    fps: float,
    scan_end_frame: int,
) -> ReacquireSeed | None:
    """Close-up seeding: no lane ladder exists, so take the DOMINANT
    swimmer (the featured athlete fills the frame). First confident big
    box seeds; identity vs a physical lane is unverified by design and the
    output is labelled CU, never merged into a lane protocol.
    """
    weights = _require_sam3_weights()
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
        return None
    stride = max(1, int(round(REACQUIRE_SEED_EVERY_SECONDS * fps)))
    frame_index = first_cut.frame_index
    seed: ReacquireSeed | None = None
    while frame_index < scan_end_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        print(f"  SAM 3 dominant scan t={frame_index / fps:.2f}s...", flush=True)
        predictor.set_image(frame)
        results = predictor(text=list(SAM3_TEXT_PROMPTS))
        result = results[0] if isinstance(results, list) else results
        boxes = _boxes_from_sam3_result(result)
        big = [
            (box, conf)
            for box, conf in boxes
            if conf >= 0.5
            and (box[2] - box[0]) * (box[3] - box[1]) >= 0.015 * w * h
        ]
        if big:
            big.sort(key=lambda bc: (bc[0][2] - bc[0][0]) * (bc[0][3] - bc[0][1]))
            box, conf = big[-1]
            seed = ReacquireSeed(
                frame_index=frame_index,
                time_seconds=frame_index / fps,
                box=tuple(int(v) for v in box),
                hits=1,
                mean_confidence=float(conf),
                ropes=(),
            )
            break
        frame_index += stride
    capture.release()
    return seed


def _lane_tracked_before(
    out_dir: Path,
    lane: int,
    cut_time: float,
    window: float = 2.5,
) -> bool:
    """Cross-shot prior: do box CSVs on disk prove this lane was being
    TRACKED within ``window`` seconds before ``cut_time``? Runs are separate
    CLI invocations, so the disk is the only state that survives."""
    import csv as _csv

    hits = 0
    for path in out_dir.glob(f"lane{lane}_*_boxes.csv"):
        try:
            with open(path, newline="") as handle:
                for row in _csv.DictReader(handle):
                    t = float(row["time_s"])
                    if cut_time - window <= t < cut_time and row["state"] == "TRACKING":
                        hits += 1
                        if hits >= 15:  # half a second of evidence
                            return True
        except (OSError, ValueError, KeyError):
            continue
    return False


def retrack_lane_after_first_cut(
    video_path: Path,
    lane: int,
    closest_lane: int,
    preview_seconds: float | None = None,
    cut_index: int = 1,
) -> Path | None:
    """Track one lane across the whole shot that follows a selected cut.

    ``cut_index`` is 1-based (1 = first hard cut). The SAM 3 seed usually
    lands a few seconds into the shot, so SAM 2 runs twice from that seed:
    backwards to the cut, then forwards to the next cut.
    ``preview_seconds`` optionally caps the end time.
    """
    if not 1 <= lane <= LANE_COUNT:
        raise ValueError(f"lane must be in 1..{LANE_COUNT}")
    if closest_lane not in (1, LANE_COUNT):
        raise ValueError(f"closest_lane must be 1 or {LANE_COUNT}")
    if cut_index < 1:
        raise ValueError("cut_index must be >= 1")
    if preview_seconds is not None and preview_seconds <= 0:
        raise ValueError("preview_seconds must be positive")

    if preview_seconds is None:
        print("Detecting cuts in the whole video...")
    else:
        print(f"Detecting cuts before {preview_seconds:.1f}s...")
    cuts = detect_cuts(video_path, max_seconds=preview_seconds)
    if not cuts:
        print("No hard cut found; SAM 3 -> SAM 2 re-track skipped.")
        return None
    if cut_index > len(cuts):
        print(
            f"Only {len(cuts)} cut(s) found; --cut {cut_index} is out of range."
        )
        return None
    shot_cut = cuts[cut_index - 1]
    next_cut = next(
        (cut for cut in cuts[cut_index:] if cut.frame_index > shot_cut.frame_index),
        None,
    )
    shot_start = shot_cut.time_seconds
    shot_end = (
        next_cut.time_seconds
        if next_cut is not None
        else _video_duration_seconds(video_path)
    )
    if preview_seconds is not None:
        shot_end = min(shot_end, preview_seconds)
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    scan_end_frame = int(round(shot_end * fps))

    print(
        f"Cut #{cut_index}/{len(cuts)} at {shot_start:.3f}s. "
        f"Scanning lane {lane} with SAM 3 until a lock "
        f"(conf>={REACQUIRE_MIN_CONFIDENCE:.2f}, "
        f"hits>={REACQUIRE_STABLE_HITS}, every "
        f"{REACQUIRE_SEED_EVERY_SECONDS:.1f}s) or the next cut at "
        f"{shot_end:.3f}s..."
    )
    yolo_model = _optional_swimmer_yolo()
    out_dir = OUTPUT_FOLDER / video_path.stem
    close_up = _close_up_shot(video_path, shot_cut, fps, shot_end)
    if close_up:
        print(
            "  Close-up shot detected (lane ladder unsupported by real "
            "ropes) — SINGLE-LANE mode: tracking the dominant swimmer, "
            "lane identity unverified, output labelled CU."
        )
        seed = _find_dominant_seed(video_path, shot_cut, fps, scan_end_frame)
    else:
        prior = _lane_tracked_before(out_dir, lane, shot_start)
        if prior:
            print(
                f"  Cross-shot prior: lane {lane} was tracked right before this "
                "cut — a single high-confidence SAM 3 hit may seed."
            )
        seed = find_stable_lane_seed(
            video_path,
            shot_cut,
            lane,
            closest_lane,
            scan_end_frame=scan_end_frame,
            yolo_model=yolo_model,
            prior_confident=prior,
        )
    if seed is None:
        print(
            f"No stable lane-{lane} SAM 3 box appeared "
            f"{REACQUIRE_STABLE_HITS} times before {shot_end:.2f}s. "
            "SAM 2 was not started."
        )
        return None

    print(
        f"Stable seed: t={seed.time_seconds:.3f}s frame={seed.frame_index}, "
        f"hits={seed.hits}, mean confidence={seed.mean_confidence:.2f}, "
        f"box={seed.box}"
    )
    if shot_end <= seed.time_seconds + 1.0 / 30.0:
        print("The next cut/end occurs immediately after the seed; nothing to track.")
        return None

    print(
        f"Shot to cover: {shot_start:.2f}s -> {shot_end:.2f}s "
        f"({shot_end - shot_start:.2f}s); SAM 2 runs back to the cut, "
        "then forward to the next cut."
    )

    out_dir = OUTPUT_FOLDER / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    lane_label = "CU" if close_up else str(lane)
    out_path = out_dir / (
        f"{video_path.stem}_sam3_sam2_lane{lane_label}_"
        f"t{shot_start:.2f}-{shot_end:.2f}s.mp4"
    )
    if not close_up:
        _save_seed_audit(
            video_path,
            out_dir
            / (
                f"{video_path.stem}_sam3_sam2_lane{lane}_"
                f"t{shot_start:.2f}_seed.jpg"
            ),
            seed,
            lane,
            closest_lane,
        )

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        fps = 30.0

        backward_duration = seed.time_seconds - shot_start
        if backward_duration > 1.0 / 30.0:
            print(f"  Pass 1/2: back to the cut ({backward_duration:.2f}s)...")
            reversed_clip = tmp_dir / "back_source.mp4"
            reversed_raw = tmp_dir / "back_raw.mp4"
            backward_chronological = tmp_dir / "back.mp4"
            # Reversed so SAM still starts on the seed frame, then flipped back.
            _extract_clip(
                video_path,
                reversed_clip,
                shot_start,
                backward_duration,
                reverse=True,
            )
            back_result = _annotate_sam_clip(
                reversed_clip,
                reversed_raw,
                seed,
                lane,
                closest_lane,
                clip_start_time=seed.time_seconds,
                time_direction=-1,
                expected_frames=max(1, int(round(backward_duration * 30))),
                csv_dir=out_dir,
                skip_first_frame=True,
                single_lane=close_up,
                label=f"lane{lane_label}",
            )
            fps = back_result.fps
            if back_result.written > 0:
                _ffmpeg_reverse(reversed_raw, backward_chronological)
                parts.append(backward_chronological)

        forward_duration = shot_end - seed.time_seconds
        print(f"  Pass 2/2: forward to the next cut ({forward_duration:.2f}s)...")
        print(
            f"  Precomputing SAM 3 identity checks every "
            f"{REACQUIRE_VERIFY_EVERY_SECONDS:.1f}s..."
        )
        shot_verifications = () if close_up else _scan_lane_verifications(
            video_path,
            seed.time_seconds,
            shot_end,
            lane,
            closest_lane,
            seed.ropes,
        )
        current_seed = seed
        reseed_count = 0
        while current_seed.time_seconds < shot_end - 1.0 / 30.0:
            segment_duration = shot_end - current_seed.time_seconds
            segment_tag = f"fwd_{reseed_count}"
            forward_clip = tmp_dir / f"{segment_tag}_source.mp4"
            forward_raw = tmp_dir / f"{segment_tag}_raw.mp4"
            _extract_clip(
                video_path,
                forward_clip,
                current_seed.time_seconds,
                segment_duration,
            )
            forward_result = _annotate_sam_clip(
                forward_clip,
                forward_raw,
                current_seed,
                lane,
                closest_lane,
                clip_start_time=current_seed.time_seconds,
                time_direction=1,
                expected_frames=max(1, int(round(segment_duration * 30))),
                csv_dir=out_dir,
                detect_loss=not close_up,
                verifications=tuple(
                    verification
                    for verification in shot_verifications
                    if verification.time_seconds
                    > current_seed.time_seconds + 0.5 / fps
                ),
                single_lane=close_up,
                label=f"lane{lane_label}",
            )
            fps = forward_result.fps
            if forward_result.written <= 0:
                break

            lost_time = forward_result.lost_time
            if lost_time is None:
                parts.append(forward_raw)
                break
            if reseed_count >= REACQUIRE_MAX_RESEEDS:
                print(
                    f"  Mask lost at {lost_time:.2f}s; maximum "
                    f"{REACQUIRE_MAX_RESEEDS} SAM 3 re-seeds reached."
                )
                parts.append(forward_raw)
                break

            reason = forward_result.lost_reason or "tracking failure"
            print(
                f"  {reason} from {lost_time:.2f}s; asking SAM 3 to "
                f"re-acquire lane {lane}..."
            )
            lost_frame = max(
                current_seed.frame_index + 1,
                int(round(lost_time * fps)),
            )
            loss_cut = CameraCut(
                frame_index=lost_frame,
                time_seconds=lost_frame / fps,
                score=0.0,
                hist_distance=0.0,
                pixel_difference=0.0,
                edge_change=0.0,
            )
            new_seed = find_stable_lane_seed(
                video_path,
                loss_cut,
                lane,
                closest_lane,
                scan_end_frame=scan_end_frame,
                initial_ropes=forward_result.lost_ropes,
                expected_xy=forward_result.lost_xy,
                expected_time=lost_time,
                allow_weak_fallback=False,
                yolo_model=yolo_model,
            )
            if (
                new_seed is None
                or new_seed.time_seconds
                <= current_seed.time_seconds + 1.0 / fps
            ):
                print("  SAM 3 could not re-acquire this lane; keeping SAM 2 output.")
                parts.append(forward_raw)
                break

            # Keep SAM 2 only until the loss. Empty/foam frames after that
            # are replaced by a backward SAM 2 pass from the new seed.
            keep_duration = max(0.0, lost_time - current_seed.time_seconds)
            if keep_duration > 1.0 / fps:
                kept_part = tmp_dir / f"{segment_tag}_kept.mp4"
                _extract_clip(
                    forward_raw,
                    kept_part,
                    start_time=0.0,
                    duration=keep_duration,
                )
                parts.append(kept_part)

            gap_duration = new_seed.time_seconds - lost_time
            if gap_duration > 1.0 / fps:
                print(
                    f"  Filling {gap_duration:.2f}s gap backward from "
                    f"t={new_seed.time_seconds:.2f}s to t={lost_time:.2f}s..."
                )
                gap_source = tmp_dir / f"{segment_tag}_gap_source.mp4"
                gap_raw = tmp_dir / f"{segment_tag}_gap_raw.mp4"
                gap_part = tmp_dir / f"{segment_tag}_gap.mp4"
                _extract_clip(
                    video_path,
                    gap_source,
                    lost_time,
                    gap_duration,
                    reverse=True,
                )
                gap_result = _annotate_sam_clip(
                    gap_source,
                    gap_raw,
                    new_seed,
                    lane,
                    closest_lane,
                    clip_start_time=new_seed.time_seconds,
                    time_direction=-1,
                    expected_frames=max(1, int(round(gap_duration * fps))),
                    csv_dir=out_dir,
                    skip_first_frame=True,
                )
                if gap_result.written > 0:
                    _ffmpeg_reverse(gap_raw, gap_part)
                    parts.append(gap_part)

            reseed_count += 1
            print(
                f"  Re-seed #{reseed_count}: t={new_seed.time_seconds:.3f}s, "
                f"hits={new_seed.hits}, mean confidence="
                f"{new_seed.mean_confidence:.2f}, box={new_seed.box}"
            )
            _save_seed_audit(
                video_path,
                out_dir
                / (
                    f"{video_path.stem}_sam3_sam2_lane{lane}_"
                    f"t{shot_start:.2f}_reseed{reseed_count}.jpg"
                ),
                new_seed,
                lane,
                closest_lane,
            )
            current_seed = new_seed

        if not parts:
            print("SAM 2 returned no frames.")
            return None

        # The concat filter (not the demuxer) because the reversed backward
        # pass and the forward pass come out of different encoders.
        command = ["ffmpeg", "-y"]
        for part in parts:
            command += ["-i", str(part)]
        command += ["-ss", f"{shot_start:.6f}", "-i", str(video_path)]
        streams = "".join(f"[{index}:v]" for index in range(len(parts)))
        command += [
            "-filter_complex",
            f"{streams}concat=n={len(parts)}:v=1:a=0[v]",
        ]
        subprocess.run(
            command
            + [
                "-map",
                "[v]",
                "-map",
                f"{len(parts)}:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )

    log_time("SAM 3 -> SAM 2 re-track", start)
    print(f"Saved SAM 3 -> SAM 2 lane track: {out_path}")
    return out_path
