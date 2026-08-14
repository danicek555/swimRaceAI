"""swimRaceAI — reaction."""

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
from .blocks import *  # noqa: F401,F403
from .vlm import *  # noqa: F401,F403
from .tracking import *  # noqa: F401,F403


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



def confirmed_vlm_leave_time(
    labels: list[tuple[float, str]],
    need: int | None = None,
) -> float | None:
    """
    Leave = first of `need` LEFT answers in a row AFTER the last ON.

    Early false LEFT is ignored if a later frame is still ON_BLOCK.
    A lone ON between LEFTs is smoothed away first (LM flicker).
    """
    labels = smooth_vlm_isolated_on(labels)
    need = max(1, int(VLM_LEFT_CONFIRM if need is None else need))
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



def find_leave_with_vlm(
    clip_path: Path,
    track_boxes: list[tuple[int, int, int, int] | None],
    fps: float,
    beep_time: float,
    crop_dir: Path | None = None,
    samples: list[tuple[float, float, float] | None] | None = None,
) -> float | None:
    """
    Leave-block time from a vision LM.

    Samples crops from beep+0.50 to beep+1.20: every 0.03s in 0.58–0.80,
    every 0.05s elsewhere. Accepts leave only after VLM_LEFT_CONFIRM LEFT
    answers in a row.

    With VLM_TIGHT_CROP the crop window is built around the swimmer's OWN
    block (found under their feet at beep time via `samples`), other lanes
    are dimmed, and the local foot-overlap veto is anchored at those feet.
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
    reset_vlm_reference_cache()

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
    # Lock crop using the swimmer's state AT THE BEEP (still on the block),
    # not at beep+0.5s when the track box may already be tiny mid-dive.
    locked_rect: tuple[int, int, int, int] | None = None
    locked_cnt: np.ndarray | None = None
    beep_idx = int(round(beep_time * fps))
    seed_box = nearest_track_box(track_boxes, beep_idx)
    if seed_box is None:
        seed_box = nearest_track_box(track_boxes, max(0, beep_idx - 5))

    # The swimmer's feet at the beep (full-frame coords): the one anchor that
    # is guaranteed to sit on THEIR block, not on a neighbouring lane's.
    foot_xy: tuple[float, float] | None = None
    body_scale = 120.0
    if samples:
        beep_sample = nearest_frame_sample(samples, beep_idx)
        if beep_sample is not None:
            foot_xy = (beep_sample[0], beep_sample[2])
        built = build_motion_signal(samples, fps, beep_time)
        if built is not None:
            body_scale = built[2]
        elif seed_box is not None:
            body_scale = float(
                max(seed_box[2] - seed_box[0], seed_box[3] - seed_box[1], 80)
            )
    elif seed_box is not None:
        body_scale = float(
            max(seed_box[2] - seed_box[0], seed_box[3] - seed_box[1], 80)
        )

    dive_sign = dive_direction_sign(samples, fps, beep_time) if samples else 1
    if VLM_TIGHT_CROP and foot_xy is not None:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            min(max(beep_idx, 0), len(track_boxes) - 1),
        )
        ok_beep, beep_frame = capture.read()
        if ok_beep:
            sign = dive_sign
            locked_rect, locked_cnt, cnt_from_red = tight_block_crop_rect(
                beep_frame, foot_xy, body_scale, sign
            )
            if locked_rect is not None:
                lw = locked_rect[2] - locked_rect[0]
                lh = locked_rect[3] - locked_rect[1]
                print(
                    f"Locked VLM crop window {locked_rect} ({lw}x{lh}px, "
                    f"tight around the swimmer's block; dive dir "
                    f"{'right' if sign > 0 else 'left'}; block contour "
                    f"{'red' if cnt_from_red else 'synthesized from feet'})"
                )
            else:
                print("Tight block crop failed — falling back to body-box crop.")

    # Occlusion baseline: the platform shape + pixels from a frame where it is
    # EMPTY (just past the search window — the swimmer is gone by then). The
    # beep-time contour has a bite where the feet occluded the red top, and
    # the empty pixels are the reference for colour-free frame differencing.
    platform_mask: np.ndarray | None = None
    ref_crop: np.ndarray | None = None
    if locked_rect is not None:
        anchor_crop = None
        if foot_xy is not None:
            anchor_crop = (
                foot_xy[0] - locked_rect[0],
                foot_xy[1] - locked_rect[1],
            )
        ref_idx = min(
            int(round((beep_time + VLM_RT_SEARCH_SECONDS + 0.05) * fps)),
            len(track_boxes) - 1,
        )
        locked_cnt, ref_frame = empty_platform_contour(
            capture, locked_rect, anchor_crop, ref_idx, locked_cnt
        )

        # Re-lock the window from the TRUE platform. The beep-time contour is
        # often just a sliver peeking around the swimmer's body, and a window
        # built from a sliver is unusably small. The empty-reference contour
        # has the platform's real extent — rebuild the rect from it.
        if locked_cnt is not None:
            bx, by, bwc, bhc = cv2.boundingRect(locked_cnt)
            new_rect = rect_from_block_bbox(
                beep_frame.shape[:2],
                (bx + locked_rect[0], by + locked_rect[1], bwc, bhc),
                body_scale,
                dive_sign,
            )
            if new_rect is not None and new_rect != locked_rect:
                shift = np.array(
                    [locked_rect[0] - new_rect[0], locked_rect[1] - new_rect[1]],
                    dtype=np.int32,
                )
                locked_cnt = locked_cnt + shift
                locked_rect = new_rect
                lw = locked_rect[2] - locked_rect[0]
                lh = locked_rect[3] - locked_rect[1]
                print(
                    f"Re-locked crop window from the empty platform: "
                    f"{locked_rect} ({lw}x{lh}px)"
                )

        ref_crop = extract_crop(ref_frame, locked_rect) if ref_frame is not None else None
        crop_hw = (locked_rect[3] - locked_rect[1], locked_rect[2] - locked_rect[0])
        platform_mask = filled_platform_mask(locked_cnt, crop_hw)

        # Pan guard: compare beep vs reference crop on the deck above the
        # platform, ignoring the swimmer at both moments. A moving camera
        # invalidates the locked window — switch the veto off, keep the LM.
        if platform_mask is not None:
            beep_crop = extract_crop(beep_frame, locked_rect)
            exclude = cv2.dilate(platform_mask, np.ones((25, 25), np.uint8))
            for swim_box in (seed_box, nearest_track_box(track_boxes, ref_idx)):
                if swim_box is None:
                    continue
                bx1 = max(0, int(swim_box[0]) - locked_rect[0] - 8)
                by1 = max(0, int(swim_box[1]) - locked_rect[1] - 8)
                bx2 = min(crop_hw[1], int(swim_box[2]) - locked_rect[0] + 8)
                by2 = min(crop_hw[0], int(swim_box[3]) - locked_rect[1] + 8)
                if bx2 > bx1 and by2 > by1:
                    exclude[by1:by2, bx1:bx2] = 255
            platform_top = int(cv2.boundingRect(locked_cnt)[1])
            if not scene_looks_static(beep_crop, ref_crop, exclude, platform_top):
                print(
                    "Scene not static between beep and reference (camera pan?) "
                    "— occlusion veto disabled."
                )
                platform_mask = None
                ref_crop = None

        if platform_mask is not None:
            print(
                f"Occlusion veto active ({'diff' if ref_crop is not None else 'red'}-based): "
                f"platform {int(np.count_nonzero(platform_mask))}px, force ON above "
                f"{VLM_OCCLUSION_FORCE_ON_FRAC:.0%} covered"
            )

    # With the veto active two LEFTs suffice; without it demand one more.
    need_confirm = VLM_LEFT_CONFIRM if platform_mask is not None else VLM_LEFT_CONFIRM + 1
    # Platform bbox in FRAME coords for the veto guard: occlusion only counts
    # while OUR swimmer is still near the platform — a foreign foot crossing
    # the polygon in 2D must not fake a contact.
    platform_bbox: tuple[int, int, int, int] | None = None
    if locked_cnt is not None and locked_rect is not None:
        pbx, pby, pbw, pbh = cv2.boundingRect(locked_cnt)
        margin = int(0.6 * max(pbw, pbh))
        platform_bbox = (
            locked_rect[0] + pbx - margin,
            locked_rect[1] + pby - margin,
            locked_rect[0] + pbx + pbw + margin,
            locked_rect[1] + pby + pbh + margin,
        )

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
        # Local veto evidence. Preferred: platform occlusion (shadow-proof).
        # Legacy fallback: skin∩block overlap anchored at the beep-time feet.
        if platform_mask is not None:
            occl_px, occl_frac = platform_occlusion(raw, platform_mask, ref_crop)
            veto_on = occl_frac >= VLM_OCCLUSION_FORCE_ON_FRAC
            evidence = f"occl={occl_px}px({occl_frac:.0%})"
            if veto_on and platform_bbox is not None and box is not None:
                near = not (
                    box[2] < platform_bbox[0]
                    or box[0] > platform_bbox[2]
                    or box[3] < platform_bbox[1]
                    or box[1] > platform_bbox[3]
                )
                if not near:
                    veto_on = False
                    evidence += ",foreign_occluder"
        else:
            anchor = None
            if foot_xy is not None:
                ax = foot_xy[0] - locked_rect[0]
                ay = foot_xy[1] - locked_rect[1]
                if 0 <= ax < rw and 0 <= ay < rh:
                    anchor = (ax, ay)
            if anchor is None:
                anchor = (rw * 0.35, rh * 0.60)
            overlap_px, _red, cnt = foot_block_overlap_px(
                raw, anchor_xy=anchor, locked_cnt=locked_cnt
            )
            veto_on = cnt is not None and overlap_px >= VLM_OVERLAP_FORCE_ON_PX
            evidence = f"overlap={overlap_px}px"
        # Dim other lanes AFTER the overlap measurement (skin detection needs
        # clean pixels). Only when the block contour is known — otherwise the
        # dim mask could darken the swimmer's own block.
        crop = raw  # clean crop — no yellow overlay; prompt + few-shot only
        if locked_cnt is not None:
            swimmer_rect = None
            if box is not None:
                sx1 = max(0, int(box[0]) - locked_rect[0])
                sy1 = max(0, int(box[1]) - locked_rect[1])
                sx2 = min(rw, int(box[2]) - locked_rect[0])
                sy2 = min(rh, int(box[3]) - locked_rect[1])
                if sx2 > sx1 and sy2 > sy1:
                    swimmer_rect = (sx1, sy1, sx2, sy2)
            crop = dim_crop_outside_target(raw, locked_cnt, swimmer_rect)
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
        detail = f"vlm+{evidence}"
        if raw_vlm:
            one_line = " ".join(raw_vlm.split())
            if len(one_line) > 120:
                one_line = one_line[:117] + "..."
            detail += f" | {one_line}"
        if label == "LEFT_BLOCK" and veto_on:
            label = "ON_BLOCK"
            detail = f"veto_LEFT {evidence}"
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

        # LOCK on the first confirmed streak and stop sampling immediately —
        # later samples are where a neighbour's foot flies through the crop
        # and a parasitic ON would reopen the confirmation.
        leave = confirmed_vlm_leave_time(labels, need=need_confirm)
        if leave is not None:
            break

    capture.release()

    first_left = confirmed_vlm_leave_time(labels, need=need_confirm)
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



