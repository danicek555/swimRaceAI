"""First coach-facing metrics from box CSVs — no pose model needed.

Reads the ``*_boxes.csv`` segments that the re-track pipeline writes next to
each shot video and produces:

- swim speed v(t) in m/s — smoothed box center converted to meters through
  the lane geometry scale that is already baked into the CSV
  (px_per_m = smooth_width / length_m on TRACKING rows),
- stroke tempo in cycles/min — the raw box's LEADING edge oscillates with
  every arm entry while the smoothed edge follows the body; the residual
  between them is periodic, and a windowed FFT finds the dominant rate.

Usage:
    python3 analysis/speed_tempo.py output/test2/*_boxes.csv --out output/test2
    python3 analysis/speed_tempo.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SPEED_SMOOTH_SECONDS = 0.5
TEMPO_WINDOW_SECONDS = 5.0
TEMPO_STEP_SECONDS = 0.5
TEMPO_BAND_HZ = (0.45, 1.50)  # 27..90 cycles/min covers all four strokes
TEMPO_MIN_PROMINENCE = 1.6    # peak power vs median in-band power
TEMPO_MAX_NAN_FRAC = 0.5


def load_rows(paths: list[Path]) -> dict[str, np.ndarray]:
    """Merge CSV segments into time-sorted arrays (NaN where missing)."""
    rows: dict[float, dict] = {}
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                t = float(row["time_s"])
                # Segments overlap at seed frames; first writer wins.
                rows.setdefault(t, row)
    if not rows:
        raise SystemExit("no rows found in the given CSVs")
    times = np.array(sorted(rows))

    def col(name: str) -> np.ndarray:
        out = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            value = rows[t].get(name, "")
            if value not in ("", None):
                out[i] = float(value)
        return out

    states = np.array([rows[t]["state"] for t in times])
    sy1 = col("smooth_y1")
    sy2 = col("smooth_y2")
    return {
        "t": times,
        "sx1": col("smooth_x1"),
        "sx2": col("smooth_x2"),
        "rx1": col("raw_x1"),
        "rx2": col("raw_x2"),
        "sy_mid": 0.5 * (sy1 + sy2),
        "length_m": col("length_m"),
        "no_pose": col("no_pose"),
        "state": states,
    }


def rolling_mean(values: np.ndarray, k: int) -> np.ndarray:
    """NaN-tolerant centered rolling mean."""
    k = max(1, k | 1)
    out = np.full_like(values, np.nan, dtype=float)
    half = k // 2
    for i in range(len(values)):
        window = values[max(0, i - half) : i + half + 1]
        good = window[~np.isnan(window)]
        if good.size:
            out[i] = float(good.mean())
    return out


def load_camera_track(path: Path) -> dict[str, np.ndarray] | None:
    """Sidecar CSV written by swim/registration.py (t, cam_dx_px)."""
    if not path.is_file():
        return None
    import csv as _csv

    ts, dxs = [], []
    with open(path, newline="") as handle:
        for row in _csv.DictReader(handle):
            ts.append(float(row["time_s"]))
            dxs.append(float(row["cam_dx_px"]))
    if not ts:
        return None
    return {"t": np.asarray(ts), "cam_dx": np.asarray(dxs)}


def load_pool_x(path: Path, times: np.ndarray) -> np.ndarray | None:
    """Sidecar from analysis/pool_pass.py aligned onto row times."""
    if not path.is_file():
        return None
    import csv as _csv

    vals: dict[float, float] = {}
    with open(path, newline="") as handle:
        for row in _csv.DictReader(handle):
            if row["pool_x_m"]:
                vals[round(float(row["time_s"]), 3)] = float(row["pool_x_m"])
    if not vals:
        return None
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        out[i] = vals.get(round(float(t), 3), np.nan)
    return out


def compute_speed(
    data: dict[str, np.ndarray],
    fps_hint: float,
    camera: dict[str, np.ndarray] | None = None,
    pool_x: np.ndarray | None = None,
) -> dict:
    """v(t) in m/s from the smoothed center + the lane scale in the CSV.

    With a camera track, positions move to POOL-fixed x (image + camera
    pan) — speeds become absolute and turns visible even under a pan.
    """
    t = data["t"]
    cx = 0.5 * (data["sx1"] + data["sx2"])
    if camera is not None:
        cx = cx + np.interp(t, camera["t"], camera["cam_dx"])
    width = data["sx2"] - data["sx1"]

    ppm_raw = width / data["length_m"]  # px per meter where length_m exists
    good = ~np.isnan(ppm_raw)
    if good.sum() < 5:
        raise SystemExit("too few rows with length_m — cannot calibrate px/m")
    ppm = np.interp(t, t[good], ppm_raw[good])

    tracking = data["state"] == "TRACKING"

    # Re-seed seams: the box re-initialises and the center JUMPS while the
    # swimmer does not. A jump larger than 0.6 m within one frame step is a
    # coordinate step, not motion — subtract it from everything after, so
    # the velocity flows through the seam instead of being masked out.
    cx = cx.copy()
    offset = 0.0
    prev_val = None
    for i in range(len(cx)):
        if np.isnan(cx[i]):
            continue
        if prev_val is not None:
            step = cx[i] - prev_val
            if abs(step) > 0.6 * ppm[i]:
                offset += step
        prev_val = cx[i]
        cx[i] -= offset

    if pool_x is not None:
        # ABSOLUTE pool meters from the keyframe-homography sidecar: the
        # position is already in meters, so no px/m conversion happens at
        # all — speeds are absolute and turns live at real walls.
        pos = rolling_mean(
            np.where(tracking, pool_x, np.nan),
            int(SPEED_SMOOTH_SECONDS * fps_hint),
        )
        v_m = np.gradient(pos, t)
        speed = np.abs(v_m)
        speed[~tracking] = np.nan
        spike = speed > 3.0
        for shift in (-2, -1, 1, 2):
            spike |= np.roll(spike, shift)
        speed[spike] = np.nan
        dt_h = np.diff(t)
        hole = np.zeros(len(t), dtype=bool)
        hole[:-1] |= dt_h > 0.5
        hole[1:] |= dt_h > 0.5
        speed[hole] = np.nan
        v_masked = np.where(np.isnan(speed), np.nan, v_m)
        direction = np.sign(rolling_mean(v_masked, int(1.0 * fps_hint)))
        # Tempo needs the IMAGE-space direction: the leading-edge residual
        # lives in image pixels, and under a pan the image motion can point
        # the other way than the pool motion.
        cx_img = rolling_mean(
            np.where(tracking, cx, np.nan), int(SPEED_SMOOTH_SECONDS * fps_hint)
        )
        v_img = np.gradient(cx_img, t)
        direction_img = np.sign(rolling_mean(v_img, int(1.0 * fps_hint)))
        return {
            "speed": speed,
            "ppm": ppm,
            "direction": direction,
            "direction_img": direction_img,
            "cx": cx,
            "cx_m": np.where(np.isfinite(pos), pos, np.nan),
        }

    cx_smooth = rolling_mean(np.where(tracking, cx, np.nan), int(SPEED_SMOOTH_SECONDS * fps_hint))
    v_px = np.gradient(cx_smooth, t)
    speed = np.abs(v_px) / ppm
    speed[~tracking] = np.nan
    # Segment seams (re-seeds) jump the center by half a body in one frame
    # and read as impossible speeds. Nothing human swims 3 m/s mid-pool —
    # mask the spike and its neighbours instead of averaging it in.
    spike = speed > 3.0
    for shift in (-2, -1, 1, 2):
        spike |= np.roll(spike, shift)
    speed[spike] = np.nan
    # Samples next to a TIME hole (shot boundary, excluded close-up) get a
    # meaningless gradient across the hole — blank them.
    dt = np.diff(t)
    hole = np.zeros(len(t), dtype=bool)
    hole[:-1] |= dt > 0.5
    hole[1:] |= dt > 0.5
    speed[hole] = np.nan
    # Direction and integrated position must use the SAME masked velocity
    # as `speed` — the raw gradient still contains PREDICTED coasting and
    # re-seed spikes, and integrating those fabricates meters exactly where
    # the swimmer is hardest to see (turns).
    v_masked = np.where(np.isnan(speed), np.nan, v_px)
    direction = np.sign(rolling_mean(v_masked, int(1.0 * fps_hint)))
    # Pool position in METERS by integrating displacements with the LOCAL
    # scale. Dividing an absolute pixel position by local px/m is invalid:
    # with camera offsets of +-2000 px, every ppm wobble fabricates meters.
    dt = np.gradient(t)
    step_m = np.where(np.isnan(v_masked), 0.0, v_px / ppm * dt)
    cx_m = np.cumsum(step_m)
    return {
        "speed": speed,
        "ppm": ppm,
        "direction": direction,
        "cx": cx,
        "cx_m": cx_m,
    }


def compute_tempo(data: dict[str, np.ndarray], direction: np.ndarray) -> dict:
    """Windowed FFT of the leading-edge residual -> cycles/min over time."""
    t = data["t"]
    lead_raw = np.where(direction >= 0, data["rx2"], data["rx1"])
    lead_smooth = np.where(direction >= 0, data["sx2"], data["sx1"])
    residual = lead_raw - lead_smooth
    residual[data["state"] != "TRACKING"] = np.nan

    if len(t) > 1:
        fs = 1.0 / float(np.median(np.diff(t)))
    else:
        fs = 30.0
    win_n = int(TEMPO_WINDOW_SECONDS * fs)
    step_n = max(1, int(TEMPO_STEP_SECONDS * fs))
    times, rates = [], []
    for start in range(0, len(t) - win_n, step_n):
        seg = residual[start : start + win_n]
        if np.isnan(seg).mean() > TEMPO_MAX_NAN_FRAC:
            continue
        seg = seg - np.nanmean(seg)
        seg = np.nan_to_num(seg)
        seg = seg * np.hanning(len(seg))
        # Zero-pad 8x: a 6 s window alone has ~10 cycles/min bins — too
        # coarse. Padding interpolates the spectrum to ~1.3 cycles/min.
        n_fft = 8 * len(seg)
        spectrum = np.abs(np.fft.rfft(seg, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        band = (freqs >= TEMPO_BAND_HZ[0]) & (freqs <= TEMPO_BAND_HZ[1])
        if not band.any():
            continue
        band_power = spectrum[band]
        peak_idx = int(np.argmax(band_power))
        if band_power[peak_idx] < TEMPO_MIN_PROMINENCE * max(np.median(band_power), 1e-9):
            continue
        times.append(float(t[start + win_n // 2]))
        rates.append(float(freqs[band][peak_idx] * 60.0))
    return {"t": np.array(times), "cycles_per_min": np.array(rates)}


def poseable_mask(
    t: np.ndarray,
    tempo: dict,
) -> np.ndarray:
    """Per-frame pose-zone flag from PASSED tempo windows.

    A tempo window that passed the FFT prominence gate is a *certificate*
    that rhythmic stroking covers [center - win/2, center + win/2]. That is
    a far stronger periodicity test than any per-frame score: an offline
    calibration showed per-frame autocorrelation cannot separate stroking
    from turns on far lanes (medians 0.21 vs 0.09), while the window gate
    validated tempo across the whole race.
    """
    half = 0.5 * TEMPO_WINDOW_SECONDS
    mask = np.zeros(len(t), dtype=bool)
    for tc in tempo["t"]:
        mask |= (t >= tc - half) & (t <= tc + half)
    return mask


def detect_events(data: dict[str, np.ndarray], speed: dict) -> dict:
    """Race events from signals already in the CSV.

    - turn: the center's direction of travel flips and STAYS flipped,
      refined to the local speed minimum (the wall touch),
    - swimming: the raw leading edge oscillates (someone is stroking) —
      rolling std of the raw-vs-smooth residual, in meters via px/m,
    - breakout: first sustained swimming after a turn; the span between
      turn and breakout is the underwater phase.
    """
    t = data["t"]
    if len(t) < 10:
        return {"turns": [], "underwater": [], "swimming": None}
    fs = 1.0 / float(np.median(np.diff(t)))
    direction = speed["direction"]

    # --- turns: sustained direction flips ---
    turns: list[float] = []
    d = direction.copy()
    valid = ~np.isnan(d) & (d != 0)
    last_sign = 0.0
    hold = int(2.0 * fs)
    i = 0
    while i < len(t):
        if valid[i]:
            s = d[i]
            if last_sign == 0.0:
                last_sign = s
            elif s != last_sign:
                ahead = d[i : i + hold]
                ok = ahead[~np.isnan(ahead)]
                if ok.size >= hold // 3 and (ok == s).mean() > 0.8:
                    # A real wall turn is a trajectory EXTREME with real
                    # travel on both sides; slow mid-pool drift also flips
                    # the direction sign but moves centimeters, not meters.
                    ctx = int(3.0 * fs)
                    if i - ctx < 0 or i + ctx > len(t):
                        # Cannot verify a turn at the data edge; a re-seed
                        # seam there fakes the reversal too easily.
                        last_sign = s
                        i += 1
                        continue
                    cx_m = speed["cx_m"]
                    before = cx_m[max(0, i - ctx) : i + 1]
                    after = cx_m[i : i + ctx]
                    before = before[~np.isnan(before)]
                    after = after[~np.isnan(after)]
                    # NET displacement, not range: stroke surge oscillates
                    # over a meter without going anywhere. A wall turn means
                    # net travel one way, then net travel the OTHER way.
                    travel_ok = False
                    if before.size > 3 and after.size > 3:
                        net_b = before[-1] - before[0]
                        net_a = after[-1] - after[0]
                        travel_ok = (
                            abs(net_b) >= 1.2
                            and abs(net_a) >= 1.2
                            and np.sign(net_b) != np.sign(net_a)
                        )
                    if travel_ok:
                        # A camera pan also reverses image motion — but at
                        # image speeds no human swims. Reject flips whose
                        # median speed just after is beyond human pace.
                        chk = speed["speed"][i : i + int(2.0 * fs)]
                        chk = chk[~np.isnan(chk)]
                        if chk.size > 5 and float(np.median(chk)) > 2.6:
                            travel_ok = False
                    if travel_ok:
                        lo = max(0, i - int(1.5 * fs))
                        hi = min(len(t), i + int(1.5 * fs))
                        seg = speed["speed"][lo:hi]
                        if np.any(~np.isnan(seg)):
                            turn_i = lo + int(np.nanargmin(seg))
                        else:
                            turn_i = i
                        turns.append(float(t[turn_i]))
                        last_sign = s
                        i += hold
                        continue
                    last_sign = s
        i += 1

    # A wall turn flips the direction back and forth while the body
    # rotates — collapse flip clusters within 5 s into ONE turn at the
    # speed minimum of the cluster window.
    if turns:
        merged: list[float] = []
        cluster = [turns[0]]
        for turn_t in turns[1:]:
            if turn_t - cluster[-1] < 5.0:
                cluster.append(turn_t)
            else:
                merged.append(cluster)
                cluster = [turn_t]
        merged.append(cluster)
        refined = []
        for cl in merged:
            lo = int(np.searchsorted(t, cl[0] - 1.0))
            hi = int(np.searchsorted(t, cl[-1] + 1.0))
            seg = speed["speed"][lo:hi]
            if np.any(~np.isnan(seg)):
                refined.append(float(t[lo + int(np.nanargmin(seg))]))
            else:
                refined.append(float(np.mean(cl)))
        turns = refined

    # No two wall turns within 10 s (a pool length takes longer than that);
    # keep the one at the deeper speed minimum.
    if len(turns) > 1:
        kept: list[float] = []
        for turn_t in turns:
            if kept and turn_t - kept[-1] < 10.0:
                prev_i = int(np.searchsorted(t, kept[-1]))
                cur_i = int(np.searchsorted(t, turn_t))
                pv = speed["speed"][max(0, prev_i - 3) : prev_i + 4]
                cv = speed["speed"][max(0, cur_i - 3) : cur_i + 4]
                pv_min = np.nanmin(pv) if np.any(~np.isnan(pv)) else np.inf
                cv_min = np.nanmin(cv) if np.any(~np.isnan(cv)) else np.inf
                if cv_min < pv_min:
                    kept[-1] = turn_t
            else:
                kept.append(turn_t)
        turns = kept

    # --- underwater span: after the turn the tracker sees no surface
    # body (PREDICTED) or the crop is flagged no_pose; the span ends at the
    # first sustained clean TRACKING (the breakout). Both signals already
    # exist in the CSV — no new detector needed for v1.
    underwater: list[tuple[float, float]] = []
    hidden = (data["state"] != "TRACKING") | (data["no_pose"] == 1)
    need = int(1.0 * fs)
    for turn_t in turns:
        start_i = int(np.searchsorted(t, turn_t))
        breakout = None
        run = 0
        for j in range(start_i, len(t)):
            if not hidden[j]:
                run += 1
                if run >= need:
                    breakout = float(t[j - need + 1])
                    break
            else:
                run = 0
        if breakout is not None and breakout - turn_t > 0.4:
            underwater.append((turn_t, breakout))
    return {"turns": turns, "underwater": underwater}


def render(data, speed, tempo, out_png: Path, title: str, events: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#333333", "#666666", "#e3e3e3"
    blue = "#4269d0"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, dpi=150,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )
    fig.patch.set_facecolor("white")

    t = data["t"]
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, color=grid, linewidth=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)

    # Gray zones: PREDICTED (coasting) and NO-POSE (underwater/glide).
    def shade(ax):
        pred = data["state"] == "PREDICTED"
        nop = data["no_pose"] == 1
        for mask, color in ((pred, "#d9d9d9"), (nop, "#efefef")):
            if not mask.any():
                continue
            edges = np.flatnonzero(np.diff(mask.astype(int)))
            starts = list(edges[mask[edges + 1]] + 1) + ([0] if mask[0] else [])
            ends = list(edges[~mask[edges + 1]] + 1) + ([len(mask)] if mask[-1] else [])
            for a, b in zip(sorted(starts), sorted(ends)):
                ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, zorder=0)

    shade(ax1)
    shade(ax2)

    # Measured solid; short gaps (<=1 s) bridged dashed so the eye can
    # follow the curve while the honesty of the measurement is preserved.
    v = speed["speed"]
    filled = v.copy()
    good_idx = np.flatnonzero(~np.isnan(v))
    if good_idx.size >= 2:
        filled = np.interp(t, t[good_idx], v[good_idx])
        long_gap = np.isnan(v).copy()
        # keep long gaps (>1 s) unfilled
        i = 0
        while i < len(v):
            if np.isnan(v[i]):
                j = i
                while j < len(v) and np.isnan(v[j]):
                    j += 1
                if t[min(j, len(t) - 1)] - t[max(i - 1, 0)] <= 1.0:
                    long_gap[i:j] = False
                i = j
            else:
                i += 1
        filled[long_gap] = np.nan
    dashed = np.where(np.isnan(v), filled, np.nan)
    ax1.plot(t, dashed, color=blue, linewidth=1.4, linestyle="--", alpha=0.6)
    ax1.plot(t, v, color=blue, linewidth=2)
    good = speed["speed"][~np.isnan(speed["speed"])]
    if good.size:
        mean_v = float(good.mean())
        ax1.axhline(mean_v, color=muted, linewidth=1, linestyle=":")
        ax1.annotate(
            f"mean {mean_v:.2f} m/s",
            xy=(t[0], mean_v), xytext=(4, 5), textcoords="offset points",
            fontsize=9, color=ink,
        )
    if events:
        for turn_t in events["turns"]:
            for ax in (ax1, ax2):
                ax.axvline(turn_t, color="#444444", linewidth=1.2, linestyle="--")
            ax1.annotate(
                "obratka", xy=(turn_t, ax1.get_ylim()[1]),
                xytext=(3, -12), textcoords="offset points",
                fontsize=9, color=ink,
            )
        for a, b in events["underwater"]:
            ax1.axvspan(a, b, facecolor="none", edgecolor="#9aa7c7",
                        hatch="//", linewidth=0.0)
            ax1.annotate(
                "pod vodou", xy=(0.5 * (a + b), 0.06), xycoords=("data", "axes fraction"),
                ha="center", fontsize=8, color=muted,
            )
            ax1.axvline(b, color="#4269d0", linewidth=1.0, linestyle=":")
            ax1.annotate(
                "vyplav", xy=(b, 0.14), xycoords=("data", "axes fraction"),
                fontsize=8, color=ink, rotation=90, va="bottom",
            )
    ax1.set_ylabel("speed (m/s)", color=ink, fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.set_title(title, color=ink, fontsize=12, loc="left")

    if tempo["t"].size:
        ax2.plot(tempo["t"], tempo["cycles_per_min"], color=blue, linewidth=2,
                 marker="o", markersize=4)
    ax2.set_ylabel("stroke tempo (cycles/min)", color=ink, fontsize=10)
    ax2.set_xlabel("time in shot (s)", color=ink, fontsize=10)

    fig.savefig(out_png, bbox_inches="tight")
    print(f"chart: {out_png}")


def selftest() -> None:
    """Synthetic CSV with known truth: 1.5 m/s and 54 cycles/min."""
    import tempfile

    fps, ppm, speed_ms, cyc_hz = 30.0, 40.0, 1.5, 0.9
    rows = ["time_s,raw_x1,raw_y1,raw_x2,raw_y2,smooth_x1,smooth_y1,"
            "smooth_x2,smooth_y2,state,no_pose,length_m"]
    for i in range(int(12 * fps)):
        t = i / fps
        cx = 200 + speed_ms * ppm * t
        w = 2.0 * ppm
        osc = 12.0 * np.sin(2 * np.pi * cyc_hz * t)
        rows.append(
            f"{t:.3f},{cx - w / 2:.0f},600,{cx + w / 2 + osc:.0f},680,"
            f"{cx - w / 2:.0f},600,{cx + w / 2:.0f},680,TRACKING,0,2.00"
        )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "seg_boxes.csv"
        p.write_text("\n".join(rows) + "\n")
        data = load_rows([p])
        sp = compute_speed(data, fps)
        te = compute_tempo(data, sp["direction"])
        v = float(np.nanmean(sp["speed"]))
        r = float(np.median(te["cycles_per_min"])) if te["t"].size else float("nan")
        print(f"selftest: speed {v:.3f} m/s (truth 1.500), tempo {r:.1f} cyc/min (truth 54.0)")
        assert abs(v - speed_ms) < 0.05, "speed mismatch"
        assert abs(r - cyc_hz * 60) < 3.0, "tempo mismatch"
        print("SELFTEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed + stroke tempo from box CSVs")
    parser.add_argument("csvs", nargs="*", help="*_boxes.csv segments of ONE shot")
    parser.add_argument("--out", default=".", help="output directory for the chart")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--split-at",
        default="",
        help="comma-separated camera-cut times; blocks are split there too",
    )
    parser.add_argument(
        "--camera",
        default="",
        help="camera_track.csv from swim/registration.py (pool-fixed x)",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.csvs:
        parser.error("give at least one *_boxes.csv (or --selftest)")

    paths = [Path(p) for p in args.csvs]
    data = load_rows(paths)

    # Shots are stitched with TIME HOLES between them; cross-shot stitching
    # fakes reversals (reframing) — so compute speed/tempo/events PER
    # contiguous block and concatenate for display.
    t_all = data["t"]
    camera_track = load_camera_track(Path(args.camera)) if args.camera else None
    if camera_track is not None:
        print(f"camera track: {len(camera_track['t'])} snimku (pool-fixed x)")
    split_times = [float(x) for x in args.split_at.split(",") if x.strip()]
    cut_idx = {int(np.searchsorted(t_all, ct)) for ct in split_times}
    boundaries = sorted(
        {0, len(t_all)}
        | {i + 1 for i in range(len(t_all) - 1) if t_all[i + 1] - t_all[i] > 1.5}
        | {i for i in cut_idx if 0 < i < len(t_all)}
    )

    def slice_data(a: int, b: int) -> dict[str, np.ndarray]:
        return {k: v[a:b] for k, v in data.items()}

    sp = None
    te = {"t": np.array([]), "cycles_per_min": np.array([])}
    all_events: dict = {"turns": [], "underwater": []}
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a < 20:
            continue
        block = slice_data(a, b)
        bsp = compute_speed(block, args.fps, camera=camera_track)
        bte = compute_tempo(block, bsp["direction"])
        bev = detect_events(block, bsp)
        all_events["turns"] += bev["turns"]
        all_events["underwater"] += bev["underwater"]
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
    keep_idx = np.concatenate([
        np.arange(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])
        if b - a >= 20
    ])
    data = {k: v[keep_idx] for k, v in data.items()}
    sp = {k: np.concatenate(v) for k, v in sp.items()}

    good = sp["speed"][~np.isnan(sp["speed"])]
    print(f"rows: {len(data['t'])}  span {data['t'][0]:.2f}-{data['t'][-1]:.2f}s")
    if good.size:
        print(f"speed: mean {good.mean():.2f} m/s, p95 {np.percentile(good, 95):.2f} m/s")
    if te["t"].size:
        print(f"tempo: median {np.median(te['cycles_per_min']):.1f} cycles/min "
              f"({te['t'].size} windows)")
    else:
        print("tempo: no confident windows")

    poseable = poseable_mask(data["t"], te)
    # A 5 s certificate bridges short turns — subtract detected events and
    # non-tracking states; those are never pose material.
    poseable &= data["state"] == "TRACKING"
    for turn_t in all_events["turns"]:
        poseable &= ~((data["t"] >= turn_t - 1.5) & (data["t"] <= turn_t + 1.5))
    for a, b in all_events["underwater"]:
        poseable &= ~((data["t"] >= a) & (data["t"] <= b))
    old_flag = data["no_pose"] == 1
    n = len(data["t"])
    print(
        f"pose zones: poseable {100*poseable.mean():.0f}% of frames "
        f"(appearance flag said no_pose {100*old_flag.mean():.0f}%)"
    )
    zones_path = Path(args.out) / "pose_zones.csv"
    with open(zones_path, "w") as zf:
        zf.write("time_s,poseable,glide_appearance\n")
        for i in range(n):
            zf.write(
                f"{data['t'][i]:.3f},{int(poseable[i])},{int(old_flag[i])}\n"
            )
    print(f"zones: {zones_path}")

    events = all_events
    for turn_t in events["turns"]:
        print(f"event: obratka @ {turn_t:.2f}s")
    for a, b in events["underwater"]:
        print(f"event: pod vodou {a:.2f}-{b:.2f}s (vyplav @ {b:.2f}s, {b-a:.1f}s)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = paths[0].stem.replace("_boxes", "")
    render(data, sp, te, out_dir / f"{stem}_speed_tempo.png", stem, events=events)


if __name__ == "__main__":
    main()
