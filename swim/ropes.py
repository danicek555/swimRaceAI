"""Detect lane boundaries from lane ropes plus the water surface itself.

A naive "long coloured line" detector also fires on deck edges, gutter tiles
and scoreboard graphics, which silently shifts every lane number. The fix is
to use the water:

* an INTERIOR lane rope has water on BOTH sides,
* the outermost boundaries (far wall, near gutter) have water on ONE side,
* every accepted lane band must itself be mostly water.

For ``lane_count`` lanes this yields ``lane_count + 1`` boundaries ordered from
the top of the image to the bottom.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from .config import (
    ROPE_FITNESS_OVERRIDE_CONFIRMATIONS,
    ROPE_FITNESS_OVERRIDE_MARGIN,
    ROPE_MAX_JUMP_LANE_FRAC,
    ROPE_MAX_MISSING_UPDATES,
    ROPE_MIN_QUALITY,
    ROPE_NEW_GEOMETRY_CONFIRMATIONS,
    ROPE_SMOOTH_ALPHA,
)


class RopeLine(NamedTuple):
    """Image line represented by y = slope*x + intercept."""

    slope: float
    intercept: float
    score: float

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept


class LaneGeometry(NamedTuple):
    """A fitted lane ladder plus evidence supporting that fit."""

    ropes: list[RopeLine]
    quality: float
    inlier_count: int
    direct_matches: int
    min_water: float
    mean_water: float
    fit_error: float


class RopeTemporalFilter:
    """Smooth valid rope fits without assuming that the camera is static.

    Gradual motion is blended immediately. A large jump must repeat before it
    replaces the current geometry, which removes isolated bad ladder fits but
    still follows a genuine pan, zoom, or abrupt in-shot camera adjustment.
    """

    def __init__(self) -> None:
        self.current: list[RopeLine] | None = None
        self.pending: list[RopeLine] | None = None
        self.pending_hits = 0
        self.override: list[RopeLine] | None = None
        self.override_hits = 0
        self.missing_updates = 0

    @staticmethod
    def _distance(
        first: list[RopeLine],
        second: list[RopeLine],
        frame_width: int,
    ) -> float:
        if len(first) != len(second) or len(first) < 2:
            return float("inf")
        x_positions = (0.0, 0.5 * max(frame_width - 1, 1), frame_width - 1.0)
        shifts = [
            abs(a.y_at(x) - b.y_at(x))
            for a, b in zip(first, second)
            for x in x_positions
        ]
        mid_x = x_positions[1]
        spacings = [
            abs(first[index + 1].y_at(mid_x) - first[index].y_at(mid_x))
            for index in range(len(first) - 1)
        ]
        typical_spacing = max(float(np.median(spacings)), 1.0)
        return float(np.median(shifts)) / typical_spacing

    @staticmethod
    def _blend(
        old: list[RopeLine],
        new: list[RopeLine],
    ) -> list[RopeLine]:
        alpha = ROPE_SMOOTH_ALPHA
        return [
            RopeLine(
                slope=(1.0 - alpha) * before.slope + alpha * after.slope,
                intercept=(
                    (1.0 - alpha) * before.intercept + alpha * after.intercept
                ),
                score=after.score,
            )
            for before, after in zip(old, new)
        ]

    @classmethod
    def _looks_like_lane_index_shift(
        cls,
        current: list[RopeLine],
        detected: list[RopeLine],
        frame_width: int,
        same_index_distance: float,
    ) -> bool:
        """Detect a valid-looking ladder relabelled one or two lanes off.

        This failure is especially dangerous: repeated foam/edge fits can be
        internally consistent, so ordinary temporal confirmation eventually
        accepts them. If new L2 geometrically matches old L3 (or vice versa),
        it is an indexing error rather than plausible camera motion.
        """
        if same_index_distance <= ROPE_MAX_JUMP_LANE_FRAC:
            return False
        for offset in (-2, -1, 1, 2):
            if offset > 0:
                old_subset = current[:-offset]
                new_subset = detected[offset:]
            else:
                old_subset = current[-offset:]
                new_subset = detected[:offset]
            shifted_distance = cls._distance(
                old_subset,
                new_subset,
                frame_width,
            )
            if (
                shifted_distance <= ROPE_MAX_JUMP_LANE_FRAC
                and shifted_distance < 0.35 * same_index_distance
            ):
                return True
        return False

    def update(
        self,
        detected: list[RopeLine],
        frame_width: int,
        frame: np.ndarray | None = None,
    ) -> list[RopeLine] | None:
        """Return stable current geometry, or ``None`` after a long outage."""
        if not detected:
            self.missing_updates += 1
            if self.missing_updates > ROPE_MAX_MISSING_UPDATES:
                self.current = None
                self.pending = None
                self.pending_hits = 0
            return self.current

        self.missing_updates = 0
        if self.current is None:
            self.current = list(detected)
            return self.current

        jump = self._distance(self.current, detected, frame_width)
        if jump <= ROPE_MAX_JUMP_LANE_FRAC:
            self.current = self._blend(self.current, detected)
            self.pending = None
            self.pending_hits = 0
            return self.current

        # Held geometry is never trusted on faith: if it explains this frame
        # worse than the candidate does, it is replaced immediately. Without
        # this escape hatch one bad fit could survive the whole shot.
        if frame is not None:
            evidence = frame_lane_evidence(frame)
            held_fitness = lane_geometry_fitness(frame, self.current, evidence)
            candidate_fitness = lane_geometry_fitness(frame, detected, evidence)
            if candidate_fitness > held_fitness + ROPE_FITNESS_OVERRIDE_MARGIN:
                # Correction still needs agreement across updates, otherwise the
                # overlay flips between two similarly scored interpretations.
                if self.override is not None and self._distance(
                    self.override, detected, frame_width
                ) <= ROPE_MAX_JUMP_LANE_FRAC:
                    self.override_hits += 1
                else:
                    self.override = list(detected)
                    self.override_hits = 1
                if self.override_hits >= ROPE_FITNESS_OVERRIDE_CONFIRMATIONS:
                    self.current = list(detected)
                    self.override = None
                    self.override_hits = 0
                    self.pending = None
                    self.pending_hits = 0
                return self.current
            self.override = None
            self.override_hits = 0
            if held_fitness >= candidate_fitness:
                self.pending = None
                self.pending_hits = 0
                return self.current

        if self._looks_like_lane_index_shift(
            self.current,
            detected,
            frame_width,
            jump,
        ):
            self.pending = None
            self.pending_hits = 0
            return self.current

        # Repeated similar results confirm camera motion rather than one bad
        # fit caused by foam, an underwater swimmer, or occlusion.
        if (
            self.pending is not None
            and self._distance(self.pending, detected, frame_width)
            <= ROPE_MAX_JUMP_LANE_FRAC
        ):
            self.pending_hits += 1
        else:
            self.pending = list(detected)
            self.pending_hits = 1
        if self.pending_hits >= ROPE_NEW_GEOMETRY_CONFIRMATIONS:
            self.current = list(detected)
            self.pending = None
            self.pending_hits = 0
        return self.current


def raw_water_mask(frame: np.ndarray) -> np.ndarray:
    """Per-pixel water/foam mask (no morphology)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    water = (hue > 75) & (hue < 105) & (saturation > 60) & (value > 60)
    foam = (saturation < 70) & (value > 170)
    return (water | foam).astype(np.uint8)


def pool_water_mask(frame: np.ndarray) -> np.ndarray:
    """Filled swimming-surface mask used to validate lane bands."""
    mask = raw_water_mask(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pool = (labels == largest).astype(np.uint8)
    # Close holes punched by swimmers/ropes so "water in this lane?" is about
    # the pool surface, not splash gaps.
    pool = cv2.morphologyEx(pool, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return pool * 255


def _water_fraction(
    pool: np.ndarray,
    line: RopeLine,
    y_offset: float,
) -> float:
    """Fraction of sampled points that are water at a fixed offset from a line."""
    height, width = pool.shape[:2]
    xs = np.linspace(0, width - 1, 60).astype(int)
    ys = np.clip((line.slope * xs + line.intercept + y_offset).astype(int), 0, height - 1)
    return float(np.mean(pool[ys, xs] > 0))


def _band_water_fraction(
    pool: np.ndarray,
    upper: RopeLine,
    lower: RopeLine,
) -> float:
    """Fraction of water *inside* the band between two boundaries.

    This is the core "is there water in this lane?" test. A valid lane is a
    water strip; a deck/gutter strip between two false lines fails this.
    """
    height, width = pool.shape[:2]
    samples: list[float] = []
    for x in np.linspace(0, width - 1, 40).astype(int):
        y0, y1 = sorted(
            (int(round(upper.y_at(x))), int(round(lower.y_at(x))))
        )
        # Clip only after ordering. Clipping first can swap an off-screen band
        # into an inverted range, which samples nothing and yields NaN.
        y0 = max(y0, 0)
        y1 = min(y1, height - 1)
        if y1 - y0 < 3:
            continue
        column = pool[y0:y1, x]
        samples.append(float(np.mean(column > 0)))
    if not samples:
        return 0.0
    return float(np.mean(samples))


def _edge_anchor_gaps(
    pool: np.ndarray,
    boundaries: list[RopeLine],
) -> tuple[float, float] | None:
    """Distance from the outer rungs to the water edges, per ladder span.

    Positive values mean the ladder overhangs the pool: the top rung above the
    far water edge, or the bottom rung below the near edge. Normalising by the
    whole span rather than by the local lane width keeps the test meaningful at
    the top of the frame, where perspective squeezes lanes to a few pixels.
    """
    top_edge = _water_edge_line(pool, top=True)
    bottom_edge = _water_edge_line(pool, top=False)
    if top_edge is None or bottom_edge is None or len(boundaries) < 2:
        return None
    mid_x = pool.shape[1] / 2
    rows = [line.y_at(mid_x) for line in boundaries]
    span = max(rows[-1] - rows[0], 1.0)
    top_gap = (top_edge.y_at(mid_x) - rows[0]) / span
    bottom_gap = (rows[-1] - bottom_edge.y_at(mid_x)) / span
    return top_gap, bottom_gap


def _anchor_score(gaps: tuple[float, float] | None) -> float:
    """1.0 when both outer rungs sit on the water edges, 0.0 when a lane is off."""
    if gaps is None:
        return 0.5
    penalty = (min(abs(gaps[0]), 0.25) + min(abs(gaps[1]), 0.25)) / 0.25
    return max(0.0, 1.0 - 0.5 * penalty)


class LaneEvidence(NamedTuple):
    """Per-frame measurements shared by every geometry being compared."""

    pool: np.ndarray
    rope_rows: list[float]


def frame_lane_evidence(frame: np.ndarray) -> LaneEvidence:
    """Measure the water surface and visible rope rows once per frame."""
    pool = pool_water_mask(frame)
    probe = max(12.0, 0.015 * frame.shape[0])
    mid_x = frame.shape[1] / 2
    rope_rows = [
        line.y_at(mid_x)
        for line in _line_candidates(frame)
        if _water_fraction(pool, line, -probe) >= 0.55
        or _water_fraction(pool, line, probe) >= 0.55
    ]
    return LaneEvidence(pool, rope_rows)


def _rope_support(rows: list[float], observed: list[float]) -> float:
    """Fraction of boundaries that land on an actually visible lane rope."""
    if not observed or len(rows) < 2:
        return 0.0
    spacings = [after - before for before, after in zip(rows, rows[1:])]
    hits = 0
    for index, row in enumerate(rows):
        local = spacings[max(0, index - 1) : min(len(spacings), index + 1)]
        tolerance = max(8.0, 0.22 * float(np.median(local or spacings)))
        if min(abs(row - value) for value in observed) <= tolerance:
            hits += 1
    return hits / float(len(rows))


def lane_geometry_fitness(
    frame: np.ndarray,
    ropes: list[RopeLine],
    evidence: LaneEvidence | None = None,
) -> float:
    """Score how well boundaries explain THIS frame.

    Lets the temporal filter compare the geometry it is holding against a new
    candidate on equal terms, so a wrong fit cannot be locked in forever.
    """
    if len(ropes) < 2:
        return 0.0
    evidence = evidence or frame_lane_evidence(frame)
    fractions = [
        _band_water_fraction(evidence.pool, upper, lower)
        for upper, lower in zip(ropes, ropes[1:])
    ]
    if not fractions:
        return 0.0
    anchor = _anchor_score(_edge_anchor_gaps(evidence.pool, ropes))
    mid_x = frame.shape[1] / 2
    support = _rope_support(
        [line.y_at(mid_x) for line in ropes],
        evidence.rope_rows,
    )
    return float(
        0.25 * float(np.mean(fractions))
        + 0.10 * float(min(fractions))
        + 0.30 * anchor
        + 0.35 * support
    )


def _line_candidates(frame: np.ndarray) -> list[RopeLine]:
    """Long, near-horizontal lines on saturated lane-rope colours."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = ((hue < 10) | (hue > 170)) & (saturation > 130) & (value > 70)
    yellow = (hue > 14) & (hue < 38) & (saturation > 145) & (value > 100)
    blue = (
        (hue > 105)
        & (hue < 140)
        & (saturation > 135)
        & (value > 45)
        & (value < 195)
    )
    mask = (red | yellow | blue).astype(np.uint8) * 255

    raw = cv2.HoughLinesP(
        cv2.Canny(mask, 30, 100),
        1,
        np.pi / 360,
        threshold=35,
        minLineLength=max(120, int(0.13 * width)),
        maxLineGap=max(50, int(0.065 * width)),
    )
    if raw is None:
        return []

    segments: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        dx = float(x2 - x1)
        if abs(dx) < 1:
            continue
        slope = float((y2 - y1) / dx)
        if not -0.24 < slope < 0.24:
            continue
        intercept = float(y1 - slope * x1)
        center_y = slope * (width / 2) + intercept
        segments.append((center_y, slope, intercept, float(np.hypot(dx, y2 - y1))))

    # A floating rope is thick, so Hough returns both of its edges. Merge them.
    clusters: list[list[tuple[float, float, float, float]]] = []
    cluster_distance = max(14.0, 0.023 * height)
    for segment in sorted(segments, key=lambda item: item[0]):
        if clusters:
            total = sum(item[3] for item in clusters[-1])
            center = sum(item[0] * item[3] for item in clusters[-1]) / total
        else:
            center = -1e9
        if clusters and abs(segment[0] - center) < cluster_distance:
            clusters[-1].append(segment)
        else:
            clusters.append([segment])

    lines: list[RopeLine] = []
    minimum_score = max(350.0, 0.20 * width)
    for cluster in clusters:
        score = sum(item[3] for item in cluster)
        if score < minimum_score:
            continue
        slope = sum(item[1] * item[3] for item in cluster) / score
        intercept = sum(item[2] * item[3] for item in cluster) / score
        lines.append(RopeLine(slope, intercept, score))
    return sorted(lines, key=lambda line: line.y_at(frame.shape[1] / 2))


def _water_edge_line(pool: np.ndarray, top: bool) -> RopeLine | None:
    """Fit a line to the top or bottom edge of the water surface."""
    height, width = pool.shape[:2]
    xs: list[int] = []
    ys: list[int] = []
    for x in range(0, width, 20):
        column = np.where(pool[:, x] > 0)[0]
        if len(column) < 30:
            continue
        xs.append(x)
        ys.append(int(column.min() if top else column.max()))
    if len(xs) < 8:
        return None
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return RopeLine(float(slope), float(intercept), score=float(len(xs)))


class _LadderFit(NamedTuple):
    """One candidate perspective ladder fitted to observed boundary rows."""

    horizon: float
    spacing: float
    anchor: float
    indices: list[int]
    inliers: list[int]
    error: float


def _ladder_fits(
    ys: list[float],
    lane_count: int,
    tolerance: float = 0.16,
    top_k: int = 14,
) -> list[_LadderFit]:
    """Fit equally spaced parallel lanes seen in perspective.

    Lane ropes are equally spaced in the world, so in the image
    ``1 / (y - horizon)`` is linear in the lane index. Fitting that recovers
    ropes the colour detector missed and ignores strays, instead of demanding
    that all of them be visible.

    Several plausible ladders are returned because the rows alone cannot
    settle the lane width when ropes are hidden; the caller resolves that
    ambiguity using the water surface.
    """
    if len(ys) < 4:
        return []
    ys = sorted(ys)
    span = max(ys[-1] - ys[0], 1.0)
    horizons = [ys[0] - span * factor for factor in
                (0.15, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.2, 3.0,
                 4.0, 6.0, 9.0, 14.0, 22.0, 40.0, 80.0)]

    scored: list[tuple[tuple[int, float], _LadderFit]] = []
    for horizon in horizons:
        values = [1.0 / (y - horizon) for y in ys]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                for steps in range(1, lane_count + 1):
                    spacing = (values[i] - values[j]) / steps
                    if spacing <= 1e-9:
                        continue
                    positions = [(values[i] - value) / spacing for value in values]
                    indices = [int(round(position)) for position in positions]
                    inliers = [
                        p
                        for p, position in enumerate(positions)
                        if abs(position - indices[p]) <= tolerance
                    ]
                    if len(inliers) < 4:
                        continue
                    used = [indices[p] for p in inliers]
                    if len(set(used)) != len(used):
                        continue
                    if max(used) - min(used) > lane_count:
                        continue
                    error = sum(
                        abs(positions[p] - indices[p]) for p in inliers
                    ) / len(inliers)
                    scored.append(
                        (
                            (len(inliers), -error),
                            _LadderFit(
                                horizon,
                                spacing,
                                values[i],
                                indices,
                                inliers,
                                error,
                            ),
                        )
                    )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [fit for _score, fit in scored[:top_k]]


def _ladder_y(horizon: float, spacing: float, anchor: float, index: int) -> float | None:
    """Image row for one rung of a fitted lane ladder."""
    value = anchor - index * spacing
    if value <= 1e-9:
        return None
    return horizon + 1.0 / value


def detect_lane_geometry(
    frame: np.ndarray,
    lane_count: int = 8,
) -> LaneGeometry | None:
    """Fit lanes and return both boundaries and measurable fit quality."""
    height, width = frame.shape[:2]
    pool = pool_water_mask(frame)
    if float(np.mean(pool > 0)) < 0.15:
        return None

    candidates = _line_candidates(frame)
    if not candidates:
        return None

    mid_x = width / 2
    probe = max(12.0, 0.015 * height)
    touching_water: list[RopeLine] = []
    for line in candidates:
        above = _water_fraction(pool, line, -probe)
        below = _water_fraction(pool, line, probe)
        if above >= 0.55 or below >= 0.55:
            touching_water.append(line)
    for edge in (
        _water_edge_line(pool, top=True),
        _water_edge_line(pool, top=False),
    ):
        if edge is not None:
            touching_water.append(edge)
    if len(touching_water) < 4:
        return None

    touching_water.sort(key=lambda line: line.y_at(mid_x))
    ys = [line.y_at(mid_x) for line in touching_water]
    fits = _ladder_fits(ys, lane_count)
    if not fits:
        return None

    best_boundaries: list[RopeLine] | None = None
    best_fractions: list[float] | None = None
    best_anchor = 0.0
    best_inliers = 0
    best_error = 1.0
    best_score = 0.0
    for fit in fits:
        # Slope varies smoothly with height, so interpolate it for rungs that
        # had to be filled in.
        inlier_ys = [ys[p] for p in fit.inliers]
        inlier_slopes = [touching_water[p].slope for p in fit.inliers]
        if len(set(inlier_ys)) >= 2:
            slope_a, slope_b = np.polyfit(inlier_ys, inlier_slopes, 1)
        else:
            slope_a, slope_b = 0.0, float(np.mean(inlier_slopes))

        def rung(index: int) -> RopeLine | None:
            y_mid = _ladder_y(fit.horizon, fit.spacing, fit.anchor, index)
            if y_mid is None or not -height < y_mid < 2 * height:
                return None
            slope = float(slope_a * y_mid + slope_b)
            return RopeLine(slope, float(y_mid - slope * mid_x), score=1.0)

        # A ladder is only defined up to a shift, and the observed rows cannot
        # settle the lane width on their own. The water surface resolves both:
        # every band must hold water and the outer rungs must meet its edges.
        lowest = min(fit.indices[p] for p in fit.inliers)
        highest = max(fit.indices[p] for p in fit.inliers)
        for first in range(highest - lane_count, lowest + 1):
            boundaries = [rung(first + step) for step in range(lane_count + 1)]
            if any(line is None for line in boundaries):
                continue
            boundaries = [line for line in boundaries if line is not None]
            fractions = [
                _band_water_fraction(pool, upper, lower)
                for upper, lower in zip(boundaries, boundaries[1:])
            ]
            if min(fractions) < 0.5:
                continue
            gaps = _edge_anchor_gaps(pool, boundaries)
            if gaps is not None and max(abs(gaps[0]), abs(gaps[1])) > 0.10:
                continue
            anchor = _anchor_score(gaps)
            inlier_score = min(1.0, len(fit.inliers) / float(lane_count + 1))
            score = (
                0.30 * float(np.mean(fractions))
                + 0.45 * anchor
                + 0.25 * inlier_score
            )
            if score > best_score:
                best_score = score
                best_anchor = anchor
                best_boundaries = boundaries
                best_fractions = fractions
                best_inliers = len(fit.inliers)
                best_error = fit.error
    if best_boundaries is None or best_fractions is None:
        return None

    # Reject crossed/collapsed ladders. The ordering must hold over the whole
    # frame, not just at the centre where the model was fitted.
    for x in (0.0, mid_x, width - 1.0):
        rows = [line.y_at(x) for line in best_boundaries]
        if any(after - before < 4.0 for before, after in zip(rows, rows[1:])):
            return None

    # Count generated boundaries that are supported by an actual coloured
    # rope/water edge. This is the key distinction between a real ladder and a
    # mathematically plausible ladder hallucinated from four noisy lines.
    generated_rows = [line.y_at(mid_x) for line in best_boundaries]
    spacings = [
        after - before
        for before, after in zip(generated_rows, generated_rows[1:])
    ]
    direct_matches = 0
    for index, row in enumerate(generated_rows):
        nearby_spacing = float(
            np.median(
                spacings[
                    max(0, index - 1) : min(len(spacings), index + 1)
                ]
                or spacings
            )
        )
        tolerance_px = max(8.0, 0.22 * nearby_spacing)
        if any(abs(row - observed) <= tolerance_px for observed in ys):
            direct_matches += 1

    inlier_score = min(1.0, best_inliers / float(lane_count + 1))
    direct_score = direct_matches / float(lane_count + 1)
    residual_score = max(0.0, 1.0 - best_error / 0.16)
    min_water = float(min(best_fractions))
    mean_water = float(np.mean(best_fractions))
    quality = (
        0.20 * inlier_score
        + 0.25 * direct_score
        + 0.15 * mean_water
        + 0.10 * min_water
        + 0.10 * residual_score
        + 0.20 * best_anchor
    )
    scored = [
        RopeLine(line.slope, line.intercept, score=quality)
        for line in best_boundaries
    ]
    return LaneGeometry(
        ropes=scored,
        quality=quality,
        inlier_count=best_inliers,
        direct_matches=direct_matches,
        min_water=min_water,
        mean_water=mean_water,
        fit_error=best_error,
    )


def detect_lane_ropes(
    frame: np.ndarray,
    lane_count: int = 8,
) -> list[RopeLine]:
    """Return trusted lane boundaries, or none when evidence is too weak."""
    geometry = detect_lane_geometry(frame, lane_count=lane_count)
    if geometry is None or geometry.quality < ROPE_MIN_QUALITY:
        return []
    return geometry.ropes


def lane_polygon(
    ropes: list[RopeLine],
    lane: int,
    frame_width: int,
    closest_lane: int,
) -> np.ndarray:
    """Return the four-point polygon for a physical lane."""
    lane_count = len(ropes) - 1
    if lane_count < 1 or not 1 <= lane <= lane_count:
        raise ValueError(f"lane must be in 1..{lane_count}")
    if closest_lane not in (1, lane_count):
        raise ValueError(f"closest_lane must be 1 or {lane_count}")
    # Top band is furthest from the camera. With lane 8 closest, top to bottom
    # reads 1..8; with lane 1 closest it reads 8..1.
    band_index = lane - 1 if closest_lane == lane_count else lane_count - lane
    top = ropes[band_index]
    bottom = ropes[band_index + 1]
    right = frame_width - 1
    return np.asarray(
        [
            [0, int(round(top.y_at(0)))],
            [right, int(round(top.y_at(right)))],
            [right, int(round(bottom.y_at(right)))],
            [0, int(round(bottom.y_at(0)))],
        ],
        dtype=np.int32,
    )


def lane_for_point(
    ropes: list[RopeLine],
    x: float,
    y: float,
    closest_lane: int,
) -> int | None:
    """Map an image point to its physical lane using rope boundaries."""
    lane_count = len(ropes) - 1
    if lane_count < 1 or closest_lane not in (1, lane_count):
        return None
    boundaries = [line.y_at(x) for line in ropes]
    for band_index in range(lane_count):
        if boundaries[band_index] <= y < boundaries[band_index + 1]:
            return (
                band_index + 1
                if closest_lane == lane_count
                else lane_count - band_index
            )
    return None


def draw_rope_overlay(
    frame: np.ndarray,
    ropes: list[RopeLine],
    target_lane: int | None = None,
    closest_lane: int = 8,
) -> np.ndarray:
    """Draw boundaries red and optionally tint the target lane polygon."""
    output = frame.copy()
    height, width = output.shape[:2]
    lane_count = len(ropes) - 1
    if target_lane is not None and lane_count >= 1:
        polygon = lane_polygon(ropes, target_lane, width, closest_lane)
        tinted = output.copy()
        cv2.fillPoly(tinted, [polygon], (0, 180, 255))
        output = cv2.addWeighted(output, 0.78, tinted, 0.22, 0)
    for index, line in enumerate(ropes):
        cv2.line(
            output,
            (0, int(round(line.y_at(0)))),
            (width - 1, int(round(line.y_at(width - 1)))),
            (0, 0, 255),
            4,
        )
        if lane_count >= 1 and index < lane_count:
            lane_number = (
                index + 1 if closest_lane == lane_count else lane_count - index
            )
            label_x = int(0.06 * width)
            mid_y = int(
                round(
                    0.5 * (line.y_at(label_x) + ropes[index + 1].y_at(label_x))
                )
            )
            cv2.putText(
                output,
                f"L{lane_number}",
                (label_x, min(max(mid_y, 24), height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255) if lane_number == target_lane else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return output
