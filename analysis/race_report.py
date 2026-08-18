"""One-command race protocol for one lane.

Discovers everything itself:
- all ``lane{N}_*_boxes.csv`` segments in the output folder,
- camera-cut times from ``cuts/frame_change_scores.csv`` (written by
  ``--detect-cuts`` / the re-track runs),
- degenerate blocks (e.g. a close-up shot where the lane ladder cannot
  fit) are auto-excluded by their TRACKING share instead of being listed
  by hand.

Usage:
    python3 analysis/race_report.py --dir output/test2 --lane 6
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

from speed_tempo import (
    compute_speed,
    compute_tempo,
    detect_events,
    load_camera_track,
    load_rows,
    poseable_mask,
    render,
)

MIN_BLOCK_TRACKING = 0.40
MIN_BLOCK_FRAMES = 20


def cut_times_from_csv(out_dir: Path) -> list[float]:
    path = out_dir / "cuts" / "frame_change_scores.csv"
    if not path.is_file():
        return []
    times = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_cut") == "1":
                times.append(float(row["time_seconds"]))
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-race protocol for one lane")
    parser.add_argument("--dir", required=True, help="output/<video_stem> folder")
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--use-camera", action="store_true",
                        help="experimental: apply camera_track.csv (mid-pool only)")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    files = sorted(glob.glob(str(out_dir / f"lane{args.lane}_*_boxes.csv")))
    if not files:
        raise SystemExit(f"no lane{args.lane}_*_boxes.csv in {out_dir}")
    print(f"{len(files)} CSV segments")

    data = load_rows([Path(p) for p in files])
    t_all = data["t"]

    # Camera correction is OPT-IN: the translation-only model with a scalar
    # px/m was proven insufficient near the walls (perspective anisotropy
    # along the pool axis erases turns instead of revealing them). It stays
    # available for mid-pool experiments, never as a silent default.
    camera = None
    if args.use_camera:
        camera = load_camera_track(out_dir / "camera_track.csv")
        if camera is not None:
            print("camera track: pool-fixed x AKTIVNI (experimentalni)")

    # Blocks: contiguous in time AND split at camera cuts (a cut is a
    # reframing — stitching across it fakes reversals).
    cut_ts = cut_times_from_csv(out_dir)
    cut_idx = {int(np.searchsorted(t_all, ct)) for ct in cut_ts}
    boundaries = sorted(
        {0, len(t_all)}
        | {i + 1 for i in range(len(t_all) - 1) if t_all[i + 1] - t_all[i] > 1.5}
        | {i for i in cut_idx if 0 < i < len(t_all)}
    )

    sp = None
    te = {"t": np.array([]), "cycles_per_min": np.array([])}
    events: dict = {"turns": [], "underwater": []}
    keep: list[np.ndarray] = []
    print(f"{'blok':>16} | {'TRACK%':>6} | {'v prumer':>8} | {'tempo':>6}")
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a < MIN_BLOCK_FRAMES:
            continue
        block = {k: v[a:b] for k, v in data.items()}
        track_frac = float((block["state"] == "TRACKING").mean())
        span = f"{block['t'][0]:.1f}-{block['t'][-1]:.1f}s"
        if track_frac < MIN_BLOCK_TRACKING:
            print(f"{span:>16} | {100*track_frac:5.0f}% | {'VYRAZEN':>8} | (degenerovany blok)")
            continue
        bsp = compute_speed(block, args.fps, camera=camera)
        bte = compute_tempo(block, bsp["direction"])
        bev = detect_events(block, bsp)
        good = bsp["speed"][~np.isnan(bsp["speed"])]
        v_mean = float(good.mean()) if good.size else float("nan")
        tempo_med = (
            float(np.median(bte["cycles_per_min"])) if bte["t"].size else float("nan")
        )
        print(f"{span:>16} | {100*track_frac:5.0f}% | {v_mean:6.2f} m/s | {tempo_med:4.0f} c/min")
        events["turns"] += bev["turns"]
        events["underwater"] += bev["underwater"]
        keep.append(np.arange(a, b))
        if sp is None:
            sp = {k: [v] for k, v in bsp.items()}
        else:
            for k, v in bsp.items():
                sp[k].append(v)
        te["t"] = np.concatenate([te["t"], bte["t"]])
        te["cycles_per_min"] = np.concatenate(
            [te["cycles_per_min"], bte["cycles_per_min"]]
        )
    if sp is None:
        raise SystemExit("no usable block")

    keep_idx = np.concatenate(keep)
    data = {k: v[keep_idx] for k, v in data.items()}
    sp = {k: np.concatenate(v) for k, v in sp.items()}

    poseable = poseable_mask(data["t"], te)
    poseable &= data["state"] == "TRACKING"
    for turn_t in events["turns"]:
        poseable &= ~((data["t"] >= turn_t - 1.5) & (data["t"] <= turn_t + 1.5))
    for a, b in events["underwater"]:
        poseable &= ~((data["t"] >= a) & (data["t"] <= b))

    for turn_t in events["turns"]:
        print(f"obratka @ {turn_t:.2f}s")
    for a, b in events["underwater"]:
        print(f"pod vodou {a:.2f}-{b:.2f}s")
    print(f"poseable: {100*poseable.mean():.0f}% snimku")

    zones = out_dir / f"race_pose_zones_lane{args.lane}.csv"
    with open(zones, "w") as zf:
        zf.write("time_s,poseable\n")
        for i in range(len(data["t"])):
            zf.write(f"{data['t'][i]:.3f},{int(poseable[i])}\n")
    png = out_dir / f"race_timeline_lane{args.lane}.png"
    render(data, sp, te, png, f"race timeline — lane {args.lane}", events=events)
    print(f"zones: {zones}")


if __name__ == "__main__":
    main()
