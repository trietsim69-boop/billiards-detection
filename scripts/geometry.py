"""Robust four-rail fitting from unordered table-dot centres.

The public ``fit_rails`` function is deliberately independent of YOLO. It accepts
the same point observations whether they came from labels or detector predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
import math
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PointObservation:
    x: float
    y: float
    confidence: float = 1.0
    source_index: int = -1


@dataclass(frozen=True)
class RailLine:
    """Normalized infinite line ``a*x + b*y + c = 0``."""

    a: float
    b: float
    c: float
    inlier_indices: tuple[int, ...]
    mean_residual_px: float
    max_residual_px: float
    support_span_px: float

    def distance(self, point: tuple[float, float]) -> float:
        x, y = point
        return abs(self.a * x + self.b * y + self.c)

    def direction(self) -> tuple[float, float]:
        return (-self.b, self.a)

    def as_dict(self) -> dict[str, object]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "inlier_indices": list(self.inlier_indices),
            "inlier_count": len(self.inlier_indices),
            "mean_residual_px": self.mean_residual_px,
            "max_residual_px": self.max_residual_px,
            "support_span_px": self.support_span_px,
        }


@dataclass(frozen=True)
class RailFitConfig:
    distance_threshold_ratio: float = 0.004
    min_inliers_per_rail: int = 3
    min_total_points: int = 12
    min_quad_area_fraction: float = 0.03
    max_quad_area_fraction: float = 0.95
    corner_margin_ratio: float = 0.25
    min_adjacent_angle_degrees: float = 12.0

    def distance_threshold_px(self, image_size: tuple[int, int]) -> float:
        width, height = image_size
        return self.distance_threshold_ratio * max(width, height)


@dataclass(frozen=True)
class RailFitResult:
    valid: bool
    points: tuple[PointObservation, ...]
    distance_threshold_px: float
    warnings: tuple[str, ...]
    rails: tuple[RailLine, ...] = ()
    corners: tuple[tuple[float, float], ...] = ()
    outlier_indices: tuple[int, ...] = ()
    quadrilateral_area_px: float | None = None
    quadrilateral_area_fraction: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "automatic_dots",
            "valid": self.valid,
            "input_dot_count": len(self.points),
            "points": [asdict(point) for point in self.points],
            "rail_lines": [rail.as_dict() for rail in self.rails],
            "corners_image_xy": [list(corner) for corner in self.corners],
            "dot_inliers": [
                list(rail.inlier_indices) for rail in self.rails
            ],
            "outlier_indices": list(self.outlier_indices),
            "quadrilateral_area_px": self.quadrilateral_area_px,
            "quadrilateral_area_fraction": self.quadrilateral_area_fraction,
            "ransac_distance_threshold_px": self.distance_threshold_px,
            "warnings": list(self.warnings),
        }


def _normalized_line_through(
    first: PointObservation, second: PointObservation
) -> tuple[float, float, float] | None:
    delta_x = second.x - first.x
    delta_y = second.y - first.y
    length = math.hypot(delta_x, delta_y)
    if length <= 1e-9:
        return None
    a = -delta_y / length
    b = delta_x / length
    c = -(a * first.x + b * first.y)
    return (a, b, c)


def _refine_line(
    points: Sequence[PointObservation], indices: Sequence[int]
) -> tuple[float, float, float] | None:
    if len(indices) < 2:
        return None
    coordinates = np.asarray(
        [[points[index].x, points[index].y] for index in indices], dtype=np.float32
    )
    direction_x, direction_y, x0, y0 = map(
        float, cv2.fitLine(coordinates, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    )
    a, b = -direction_y, direction_x
    c = -(a * x0 + b * y0)
    if a < 0 or (abs(a) <= 1e-12 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _distances(
    points: Sequence[PointObservation],
    indices: Sequence[int],
    coefficients: tuple[float, float, float],
) -> list[float]:
    a, b, c = coefficients
    return [abs(a * points[index].x + b * points[index].y + c) for index in indices]


def _support_span(
    points: Sequence[PointObservation],
    indices: Sequence[int],
    coefficients: tuple[float, float, float],
) -> float:
    if not indices:
        return 0.0
    a, b, _ = coefficients
    direction_x, direction_y = -b, a
    projections = [
        points[index].x * direction_x + points[index].y * direction_y
        for index in indices
    ]
    return max(projections) - min(projections)


def _fit_dominant_line(
    points: Sequence[PointObservation],
    candidate_indices: Sequence[int],
    distance_threshold_px: float,
    min_inliers: int,
) -> RailLine | None:
    best: tuple[tuple[float, ...], tuple[int, ...], tuple[float, float, float]] | None = None
    for left_position, left_index in enumerate(candidate_indices):
        for right_index in candidate_indices[left_position + 1 :]:
            hypothesis = _normalized_line_through(
                points[left_index], points[right_index]
            )
            if hypothesis is None:
                continue
            hypothesis_distances = _distances(
                points, candidate_indices, hypothesis
            )
            inliers = tuple(
                index
                for index, distance in zip(candidate_indices, hypothesis_distances)
                if distance <= distance_threshold_px
            )
            if len(inliers) < min_inliers:
                continue

            refined = _refine_line(points, inliers)
            if refined is None:
                continue
            for _ in range(3):
                refined_distances = _distances(
                    points, candidate_indices, refined
                )
                updated_inliers = tuple(
                    index
                    for index, distance in zip(
                        candidate_indices, refined_distances
                    )
                    if distance <= distance_threshold_px
                )
                if updated_inliers == inliers or len(updated_inliers) < min_inliers:
                    break
                inliers = updated_inliers
                next_refined = _refine_line(points, inliers)
                if next_refined is None:
                    break
                refined = next_refined

            residuals = _distances(points, inliers, refined)
            span = _support_span(points, inliers, refined)
            score = (
                float(len(inliers)),
                -float(np.mean(residuals)),
                span,
            )
            if best is None or score > best[0]:
                best = (score, inliers, refined)

    if best is None:
        return None
    _, inliers, coefficients = best
    residuals = _distances(points, inliers, coefficients)
    return RailLine(
        a=coefficients[0],
        b=coefficients[1],
        c=coefficients[2],
        inlier_indices=tuple(sorted(inliers)),
        mean_residual_px=float(np.mean(residuals)),
        max_residual_px=max(residuals),
        support_span_px=_support_span(points, inliers, coefficients),
    )


def _acute_angle(left: RailLine, right: RailLine) -> float:
    left_direction = np.asarray(left.direction(), dtype=float)
    right_direction = np.asarray(right.direction(), dtype=float)
    cosine = float(abs(np.dot(left_direction, right_direction)))
    return math.acos(max(-1.0, min(1.0, cosine)))


def _opposite_pairing(
    rails: Sequence[RailLine],
) -> tuple[tuple[int, int], tuple[int, int]]:
    pairings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    return min(
        pairings,
        key=lambda pairing: (
            _acute_angle(rails[pairing[0][0]], rails[pairing[0][1]])
            + _acute_angle(rails[pairing[1][0]], rails[pairing[1][1]]),
            max(
                _acute_angle(rails[pairing[0][0]], rails[pairing[0][1]]),
                _acute_angle(rails[pairing[1][0]], rails[pairing[1][1]]),
            ),
        ),
    )


def _intersection(
    left: RailLine, right: RailLine
) -> tuple[float, float] | None:
    determinant = left.a * right.b - right.a * left.b
    if abs(determinant) <= 1e-9:
        return None
    x = (left.b * right.c - right.b * left.c) / determinant
    y = (left.c * right.a - right.c * left.a) / determinant
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def _order_clockwise(
    corners: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    centroid_x = sum(point[0] for point in corners) / len(corners)
    centroid_y = sum(point[1] for point in corners) / len(corners)
    ordered = sorted(
        corners,
        key=lambda point: math.atan2(point[1] - centroid_y, point[0] - centroid_x),
    )
    start = min(
        range(len(ordered)), key=lambda index: ordered[index][0] + ordered[index][1]
    )
    rotated = ordered[start:] + ordered[:start]
    if _signed_polygon_area(rotated) < 0:
        rotated = [rotated[0], *reversed(rotated[1:])]
    return tuple(rotated)


def _signed_polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _rails_in_corner_order(
    rails: Sequence[RailLine], corners: Sequence[tuple[float, float]]
) -> tuple[RailLine, ...]:
    best: tuple[float, tuple[RailLine, ...]] | None = None
    for rail_order in permutations(rails):
        score = sum(
            rail_order[index].distance(corners[index])
            + rail_order[index].distance(corners[(index + 1) % 4])
            for index in range(4)
        )
        if best is None or score < best[0]:
            best = (score, tuple(rail_order))
    if best is None:
        raise RuntimeError("Could not order rails")
    return best[1]


def _adjacent_angles_degrees(rails: Sequence[RailLine]) -> list[float]:
    return [
        math.degrees(_acute_angle(rails[index], rails[(index + 1) % 4]))
        for index in range(4)
    ]


def fit_rails(
    points: Sequence[PointObservation],
    image_size: tuple[int, int],
    config: RailFitConfig | None = None,
) -> RailFitResult:
    """Fit four physical table rails to unordered dot-centre observations."""
    config = config or RailFitConfig()
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive")
    if config.distance_threshold_ratio <= 0:
        raise ValueError("distance_threshold_ratio must be positive")
    observations = tuple(points)
    distance_threshold = config.distance_threshold_px(image_size)
    warnings: list[str] = []

    if len(observations) < config.min_total_points:
        return RailFitResult(
            False,
            observations,
            distance_threshold,
            (f"insufficient_dot_points:{len(observations)}<{config.min_total_points}",),
            outlier_indices=tuple(range(len(observations))),
        )

    remaining = list(range(len(observations)))
    rails: list[RailLine] = []
    for _ in range(4):
        rail = _fit_dominant_line(
            observations,
            remaining,
            distance_threshold,
            config.min_inliers_per_rail,
        )
        if rail is None:
            break
        rails.append(rail)
        inlier_set = set(rail.inlier_indices)
        remaining = [index for index in remaining if index not in inlier_set]

    if len(rails) != 4:
        return RailFitResult(
            False,
            observations,
            distance_threshold,
            (f"four_rails_not_found:{len(rails)}/4",),
            rails=tuple(rails),
            outlier_indices=tuple(sorted(remaining)),
        )

    pairing = _opposite_pairing(rails)
    raw_corners: list[tuple[float, float]] = []
    for first_index in pairing[0]:
        for second_index in pairing[1]:
            corner = _intersection(rails[first_index], rails[second_index])
            if corner is None:
                return RailFitResult(
                    False,
                    observations,
                    distance_threshold,
                    ("parallel_adjacent_rails",),
                    rails=tuple(rails),
                    outlier_indices=tuple(sorted(remaining)),
                )
            raw_corners.append(corner)

    corners = _order_clockwise(raw_corners)
    ordered_rails = _rails_in_corner_order(rails, corners)
    area = abs(_signed_polygon_area(corners))
    area_fraction = area / (width * height)

    if not cv2.isContourConvex(np.asarray(corners, dtype=np.float32)):
        warnings.append("non_convex_quadrilateral")
    if area_fraction < config.min_quad_area_fraction:
        warnings.append(
            f"quadrilateral_too_small:{area_fraction:.4f}"
        )
    if area_fraction > config.max_quad_area_fraction:
        warnings.append(
            f"quadrilateral_too_large:{area_fraction:.4f}"
        )

    margin_x = width * config.corner_margin_ratio
    margin_y = height * config.corner_margin_ratio
    if any(
        x < -margin_x
        or x > width + margin_x
        or y < -margin_y
        or y > height + margin_y
        for x, y in corners
    ):
        warnings.append("corner_outside_allowed_margin")

    adjacent_angles = _adjacent_angles_degrees(ordered_rails)
    if min(adjacent_angles) < config.min_adjacent_angle_degrees:
        warnings.append(
            f"adjacent_rails_nearly_parallel:{min(adjacent_angles):.2f}deg"
        )

    assigned = {
        index for rail in ordered_rails for index in rail.inlier_indices
    }
    outliers = tuple(
        index for index in range(len(observations)) if index not in assigned
    )
    return RailFitResult(
        valid=not warnings,
        points=observations,
        rails=ordered_rails,
        corners=corners,
        outlier_indices=outliers,
        quadrilateral_area_px=area,
        quadrilateral_area_fraction=area_fraction,
        distance_threshold_px=distance_threshold,
        warnings=tuple(warnings),
    )
