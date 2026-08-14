"""swimRaceAI — pipeline."""

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
from .utils import *  # noqa: F401,F403
from .audio import *  # noqa: F401,F403
from .vlm import *  # noqa: F401,F403
from .blocks import *  # noqa: F401,F403
from .tracking import *  # noqa: F401,F403
from .detect import *  # noqa: F401,F403
from .reaction import *  # noqa: F401,F403
from .hands import *  # noqa: F401,F403
from .overlay import *  # noqa: F401,F403


def find_reaction_time_hybrid(
    clip_path: Path,
    track_boxes: list[tuple[int, int, int, int] | None],
    samples: list[tuple[float, float, float] | None],
    fps: float,
    beep_time: float,
    use_vlm: bool,
    body_frac: float | None = None,
    crop_dir: Path | None = None,
    hand_crop_dir: Path | None = None,
) -> tuple[float | None, float | None, str, HandEntryResult | None]:
    """
    Primary RT = vision-LM leave time. Also motion refine + hand entry.

    Returns (move_time, reaction, method_label, hand_entry_result).
    """
    hand_result: HandEntryResult | None = None
    if use_vlm:
        vlm_leave = find_leave_with_vlm(
            clip_path, track_boxes, fps, beep_time, crop_dir=crop_dir, samples=samples
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
            hand_result = find_hand_entry_with_vlm(
                clip_path,
                track_boxes,
                fps,
                vlm_leave,
                samples=samples,
                beep_time=beep_time,
                crop_dir=hand_crop_dir,
            )
            if hand_result.time is not None:
                flight = hand_result.time - vlm_leave
                if hand_result.status == "confident":
                    print(
                        f"Hand entry (confident): {hand_result.time:.2f}s "
                        f"(flight {flight:.2f}s after leave; "
                        f"{hand_result.time - beep_time:.2f}s after beep)"
                    )
                else:
                    print(
                        f"Hand entry ({hand_result.status}): "
                        f"{hand_result.t_lo:.2f}–{hand_result.t_hi:.2f}s "
                        f"(~{hand_result.time:.2f}s; flight ~{flight:.2f}s)"
                    )
            return (
                vlm_leave,
                lm_rt,
                "vision-LM (primary); motion refine + hand entry also printed",
                hand_result,
            )

    move_time, reaction = find_reaction_time(
        samples, fps, beep_time, body_frac=body_frac
    )
    return move_time, reaction, "motion only (body-frac)", None



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
        # Time is reported later, in the tracked video's own timeline
        # ("Beep in the tracked video: ..."), once the box frame is chosen.
        print("Beep found.")

    print(f"Using only the first {TRACK_SECONDS} seconds...")
    with tempfile.TemporaryDirectory() as tmp:
        clip_path = Path(tmp) / f"{video_path.stem}_first{TRACK_SECONDS}s.mp4"
        clip_first_seconds(video_path, clip_path, TRACK_SECONDS)

        total_frames, video_fps = clip_frame_count(clip_path)

        # Open the picker ~1s before the beep: swimmers are already set on
        # their blocks (easy to box the right one) and a second of stillness
        # remains before the beep for the motion baseline.
        suggest_idx = 0
        if beep_time is not None and video_fps > 0:
            suggest_idx = max(0, int(round((beep_time - 1.0) * video_fps)))
            suggest_idx = min(suggest_idx, max(total_frames - 1, 0))

        box, seed_idx = click_swimmer(clip_path, start_index=suggest_idx)
        if box is None:
            print("No box, skipping track.")
            return

        # SAM 2 applies the box prompt to the FIRST frame it sees — so the
        # tracked clip must START at the frame the box was drawn on.
        sam_source = clip_path
        if seed_idx > 0:
            sam_source = Path(tmp) / f"seed_{seed_idx}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(clip_path),
                    "-ss", f"{seed_idx / video_fps:.4f}",  # output seek = frame accurate
                    "-an",
                    str(sam_source),
                ],
                check=True,
                capture_output=True,
            )
            print(
                f"Your box frame becomes FRAME 0 of the tracked video "
                f"(source frame {seed_idx}); earlier frames are skipped."
            )
        if beep_time is not None and video_fps > 0 and seed_idx > int(beep_time * video_fps):
            print(
                "Warning: the box frame is AFTER the beep — the pre-beep "
                "stillness baseline is missing, motion refine will be rough."
            )

        # EVERYTHING below runs in the TRACKED-VIDEO time base: 0 = the frame
        # the box was drawn on. The output video starts there, so console
        # times, frame numbers, overlay times and the player position agree.
        time_base = seed_idx / video_fps if video_fps > 0 else 0.0
        beep_clip = None if beep_time is None else beep_time - time_base
        sam_frames = max(total_frames - seed_idx, 1)
        print(
            f"Tracked video: {sam_frames} frames, {video_fps:.1f} fps, "
            f"{sam_frames / video_fps:.2f}s (frame 0 = your box frame)."
        )
        if beep_clip is not None:
            print(
                f"Beep in the tracked video: {beep_clip:.2f}s "
                f"(frame {int(round(beep_clip * video_fps))})."
            )

        print(f"Tracking one swimmer inside box {box} with SAM 2...")
        print("First run downloads the SAM 2 file.")


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
        # Reusing a stale mask forever paints the swimmer where he no longer
        # is (and freezes the motion signal). Allow a short bridge only.
        stale_reuse_left = 9
        # Static red pixels (blocks, lane ropes) from the first frame. A
        # generous click box often includes a chunk of the starting block —
        # SAM then learns "swimmer + block" as one object and after the dive
        # the mask latches onto the empty block. Cutting pixels that are red
        # NOW and were red at t=0 removes the block without touching the
        # swimmer (a moving red suit is never red in both frames).
        red_static: np.ndarray | None = None
        # Where we think the swimmer is / is going (stops lane switches).
        pred_x = (box[0] + box[2]) / 2.0
        pred_y = (box[1] + box[3]) / 2.0
        vel_x = 0.0
        vel_y = 0.0
        # Jump limit scaled to the clicked swimmer's size: 90px suits ~1080p;
        # a bigger on-screen body legitimately moves more pixels per frame.
        max_jump_px = max(MAX_CENTER_JUMP_PX, 0.45 * max(box[2] - box[0], box[3] - box[1]))
        # One (cx, cy, foot_y) per frame OF THE TRACKED (seed) CLIP — index 0
        # is the box frame, matching the time base above. None = no mask.
        samples: list[tuple[float, float, float] | None] = []
        # Bounding box per frame for vision-LM crops (frames from the seed clip).
        track_boxes: list[tuple[int, int, int, int] | None] = []
        results = predictor(
            source=str(sam_source),
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
            if red_static is None:
                red_static = red_block_mask(result.orig_img)
            sample: tuple[float, float, float] | None = None
            frame_box: tuple[int, int, int, int] | None = None
            if result.masks is not None and len(result.masks) > 0:
                mask = result.masks.data[0].cpu().numpy()
                mask = cv2.resize(mask.astype(np.float32), (width, height))
                # Cut static red (block / lane ropes) out of the mask BEFORE
                # picking the blob — kills the "mask latched on the empty
                # block" failure at its root.
                block_now = cv2.bitwise_and(red_static, red_block_mask(result.orig_img))
                mask[block_now > 0] = 0.0
                # Follow the moving person, not the old click box on the blocks.
                mask = keep_one_person_mask(
                    mask,
                    (pred_x, pred_y),
                    first_area,
                    max_jump_px,
                )
                reused = False
                if np.count_nonzero(mask) == 0 and last_good_mask is not None:
                    if stale_reuse_left > 0:
                        stale_reuse_left -= 1
                        mask = last_good_mask
                        reused = True
                area = float(np.count_nonzero(mask))
                if area > 0:
                    if first_area is None:
                        first_area = area
                    colored = frame.copy()
                    colored[mask > 0.5] = (0, 180, 255)
                    frame = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)
                    ys, xs = np.where(mask > 0.5)
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    frame_box = (x1, y1, x2, y2)
                    if reused:
                        # A frozen mask must not feed the motion signal or the
                        # velocity model — it would report "no movement".
                        pred_x += vel_x
                        pred_y += vel_y
                    else:
                        last_good_mask = mask
                        stale_reuse_left = 9
                        cx = float(xs.mean())
                        cy = float(ys.mean())
                        # Bottom of body ~ feet on the block. 97th percentile,
                        # not max: a few stray mask pixels (rail, reflection)
                        # must not drag the foot anchor a lane lower.
                        foot_y = float(np.percentile(ys, 97))
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
                remaining = per_frame * (sam_frames - frame_count)
                print(
                    f"  {frame_count}/{total_frames} frames  "
                    f"({per_frame:.2f}s each)  "
                    f"elapsed {format_hms(elapsed)}  "
                    f"ETA {format_hms(remaining)}"
                )
                if frame_count == 3:
                    print(
                        f"  Estimated total time: "
                        f"{format_hms(per_frame * sam_frames)}"
                    )

        if writer is not None:
            writer.release()
        log_time(f"SAM 2 track ({frame_count} frames)", start)

        move_time = None
        reaction = None
        hand_entry = None
        calibrated_frac = None
        if beep_clip is not None:
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
                    samples, video_fps, beep_clip, known_rt
                )

            move_time, reaction, method, hand_entry = find_reaction_time_hybrid(
                sam_source,
                track_boxes,
                samples,
                video_fps,
                beep_clip,
                use_vlm=use_vlm and known_rt is None,
                body_frac=calibrated_frac,
                crop_dir=out_dir / "vlm_crops",
                hand_crop_dir=out_dir / "vlm_hand_crops",
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
                if (
                    isinstance(hand_entry, HandEntryResult)
                    and hand_entry.time is not None
                    and move_time is not None
                ):
                    if hand_entry.status == "confident":
                        print(
                            f"Hand entry (confident) at {hand_entry.time:.2f} seconds"
                        )
                    else:
                        print(
                            f"Hand entry ({hand_entry.status}) "
                            f"{hand_entry.t_lo:.2f}–{hand_entry.t_hi:.2f}s "
                            f"(~{hand_entry.time:.2f}s)"
                        )
                        print(f"  Reason: {hand_entry.detail}")
                    print(
                        f"Flight time (leave → hands): "
                        f"{hand_entry.time - move_time:.2f} seconds "
                        f"({(hand_entry.time - move_time) * 1000:.0f} ms)"
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
            beep_clip,
            move_time,
            reaction,
            hand_entry=hand_entry,
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
        print(f"The beep occurred at {beep_time:.3f} seconds (source video time)")

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



