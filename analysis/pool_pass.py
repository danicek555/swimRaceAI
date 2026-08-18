"""Generate the pool-coordinate sidecar for one lane's box CSVs.

For every camera shot: discover absolutely anchored keyframes
(rope-marker homographies), then convert each tracked box center to
POOL meters via blended propagation. Writes
``output/<stem>/pool_x_lane{N}.csv`` (time_s, pool_x_m).

Usage:
    python3 analysis/pool_pass.py --video videos/test2.mp4 \
        --dir output/test2 --lane 6
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swim.anchors import find_keyframes, pool_x_blended  # noqa: E402
from speed_tempo import load_camera_track, load_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    out_dir = Path(args.dir)
    camera = load_camera_track(out_dir / "camera_track.csv")
    if camera is None:
        raise SystemExit("camera_track.csv missing — run swim/registration first")
    cam_at = lambda t: float(np.interp(t, camera["t"], camera["cam_dx"]))

    files = sorted(glob.glob(str(out_dir / f"lane{args.lane}_*_boxes.csv")))
    data = load_rows([Path(p) for p in files])
    t_all = data["t"]

    cuts = []
    cuts_csv = out_dir / "cuts" / "frame_change_scores.csv"
    if cuts_csv.is_file():
        with open(cuts_csv, newline="") as handle:
            cuts = [
                float(r["time_seconds"])
                for r in csv.DictReader(handle)
                if r.get("is_cut") == "1"
            ]

    # Shots covered by the data: split row times at cuts and time holes.
    bounds = sorted(
        {float(t_all[0]), float(t_all[-1]) + 0.01}
        | {c for c in cuts if t_all[0] < c < t_all[-1]}
        | {
            float(t_all[i + 1])
            for i in range(len(t_all) - 1)
            if t_all[i + 1] - t_all[i] > 1.5
        }
    )

    capture = cv2.VideoCapture(args.video)
    pool_x = np.full(len(t_all), np.nan)
    for a, b in zip(bounds[:-1], bounds[1:]):
        mask = (t_all >= a) & (t_all < b)
        if mask.sum() < 20:
            continue
        print(f"shot {a:.1f}-{b:.1f}s: hledam keyframy...", flush=True)
        kfs = find_keyframes(capture, cam_at, a, b, fps=args.fps)
        print(f"  {len(kfs)} keyframu", flush=True)
        if not kfs:
            continue
        for i in np.flatnonzero(mask):
            if np.isnan(data["sx1"][i]):
                continue
            cx = 0.5 * (data["sx1"][i] + data["sx2"][i])
            sy1 = data.get("sy1")
            # smooth_y is not in load_rows; approximate with rope-band middle
            # via the box vertical center from raw columns when present.
            cy_val = None
            if "ry1" in data and not np.isnan(data["rx1"][i]):
                pass
            # boxes CSV stores smooth_y1/2 — extend load if absent
            cy_val = data["sy_mid"][i] if "sy_mid" in data else None
            if cy_val is None or np.isnan(cy_val):
                continue
            px = pool_x_blended(kfs, cam_at, float(t_all[i]), float(cx), float(cy_val))
            if px is not None:
                pool_x[i] = px

    out = out_dir / f"pool_x_lane{args.lane}.csv"
    with open(out, "w") as handle:
        handle.write("time_s,pool_x_m\n")
        for t, px in zip(t_all, pool_x):
            handle.write(f"{t:.3f},{'' if np.isnan(px) else f'{px:.3f}'}\n")
    good = np.isfinite(pool_x)
    print(f"pool_x: {good.sum()}/{len(pool_x)} snimku ({100*good.mean():.0f}%) -> {out}")


if __name__ == "__main__":
    main()
