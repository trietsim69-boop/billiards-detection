"""Perspective-normalized search regions for the four raised rail caps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from .table_localization import Point, TableLocalizationResult


@dataclass(frozen=True)
class RailRegionConfig:
    strip_long_resolution: int = 768
    strip_depth_resolution: int = 96
    inner_margin_ratio: float = 0.04
    cap_depth_ratio: float = 0.18
    min_valid_coverage: float = 0.45


@dataclass(frozen=True)
class RailRegion:
    side: int
    strip_image: np.ndarray
    source_polygon_xy: tuple[Point, ...]
    source_to_strip: np.ndarray
    strip_to_source: np.ndarray
    long_side: bool
    valid_pixel_coverage: float
    warnings: tuple[str, ...] = ()

    @property
    def expected_sight_count(self) -> int:
        return 6 if self.long_side else 3

    def source_to_strip_point(self, point: Point) -> Point:
        mapped = cv2.perspectiveTransform(
            np.asarray([[point]], dtype=np.float32), self.source_to_strip
        )[0, 0]
        return (float(mapped[0]), float(mapped[1]))

    def strip_to_source_point(self, point: Point) -> Point:
        mapped = cv2.perspectiveTransform(
            np.asarray([[point]], dtype=np.float32), self.strip_to_source
        )[0, 0]
        return (float(mapped[0]), float(mapped[1]))


def _outward_normal(start: Point, end: Point, centroid: Point) -> np.ndarray:
    edge = np.asarray([end[0] - start[0], end[1] - start[1]], dtype=float)
    length = float(np.linalg.norm(edge))
    if length <= 1e-9:
        raise ValueError("Rail edge must have non-zero length")
    normal = np.asarray([-edge[1], edge[0]], dtype=float) / length
    midpoint = (np.asarray(start) + np.asarray(end)) / 2.0
    if float(np.dot(normal, np.asarray(centroid) - midpoint)) > 0:
        normal = -normal
    return normal


def _shift(point: Point, vector: np.ndarray, distance: float) -> Point:
    shifted = np.asarray(point, dtype=float) + vector * distance
    return (float(shifted[0]), float(shifted[1]))


def extract_rail_regions(
    image: np.ndarray,
    table: TableLocalizationResult,
    config: RailRegionConfig | None = None,
) -> tuple[RailRegion, ...]:
    """Extract four traceable rail-cap strips from ordered table geometry.

    The band straddles each inner-cushion line slightly and extends outward by
    a fraction of that edge length.  Each band gets an independent perspective
    transform, making the separation between bed-plane and raised-rail search
    coordinates explicit.
    """
    config = config or RailRegionConfig()
    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image must be a non-empty 2D or 3D array")
    if not table.valid or len(table.bed_corners_xy) != 4:
        raise ValueError("A valid four-corner table localization is required")
    if config.strip_long_resolution < 8 or config.strip_depth_resolution < 4:
        raise ValueError("Rail strip resolutions are too small")
    if not 0 <= config.inner_margin_ratio < 1:
        raise ValueError("inner_margin_ratio must be in [0, 1)")
    if config.cap_depth_ratio <= 0:
        raise ValueError("cap_depth_ratio must be positive")
    if not 0.0 <= config.min_valid_coverage <= 1.0:
        raise ValueError("min_valid_coverage must be in [0, 1]")

    corners = table.bed_corners_xy
    centroid = (
        sum(point[0] for point in corners) / 4.0,
        sum(point[1] for point in corners) / 4.0,
    )
    edge_lengths = [
        math.dist(corners[index], corners[(index + 1) % 4]) for index in range(4)
    ]
    long_edge_length = max(edge_lengths[0], edge_lengths[2])
    regions: list[RailRegion] = []
    image_height, image_width = image.shape[:2]

    for side in range(4):
        start = corners[side]
        end = corners[(side + 1) % 4]
        edge_length = edge_lengths[side]
        normal = _outward_normal(start, end, centroid)
        inward = -normal
        inner = edge_length * config.inner_margin_ratio
        outward = edge_length * config.cap_depth_ratio
        polygon = (
            _shift(start, inward, inner),
            _shift(end, inward, inner),
            _shift(end, normal, outward),
            _shift(start, normal, outward),
        )
        strip_width = max(
            8,
            round(config.strip_long_resolution * edge_length / long_edge_length),
        )
        strip_size = (strip_width, config.strip_depth_resolution)
        source_polygon = np.asarray(polygon, dtype=np.float32)
        strip_polygon = np.asarray(
            [
                [0.0, 0.0],
                [strip_width - 1.0, 0.0],
                [strip_width - 1.0, config.strip_depth_resolution - 1.0],
                [0.0, config.strip_depth_resolution - 1.0],
            ],
            dtype=np.float32,
        )
        source_to_strip = cv2.getPerspectiveTransform(source_polygon, strip_polygon)
        strip_to_source = cv2.getPerspectiveTransform(strip_polygon, source_polygon)
        strip = cv2.warpPerspective(image, source_to_strip, strip_size)
        source_mask = np.full((image_height, image_width), 255, dtype=np.uint8)
        coverage_mask = cv2.warpPerspective(
            source_mask,
            source_to_strip,
            strip_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        coverage = float(np.count_nonzero(coverage_mask)) / coverage_mask.size
        warnings: list[str] = []
        if coverage < config.min_valid_coverage:
            warnings.append(f"low_valid_coverage:{coverage:.3f}")
        regions.append(
            RailRegion(
                side=side,
                strip_image=strip,
                source_polygon_xy=polygon,
                source_to_strip=source_to_strip,
                strip_to_source=strip_to_source,
                long_side=side in (0, 2),
                valid_pixel_coverage=coverage,
                warnings=tuple(warnings),
            )
        )
    return tuple(regions)
