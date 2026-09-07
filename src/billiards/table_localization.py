"""Contracts and small helpers for coarse table localization.

The detector and the rail search deliberately depend on this narrow result
object rather than on a particular table-localization algorithm.  That keeps
the first rail-first implementation usable with reviewed corners while the
automatic localizer is developed separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np


Point = tuple[float, float]
Line = tuple[float, float, float]


def _signed_area(points: Sequence[Point]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _line_through(first: Point, second: Point) -> Line:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("Table corners must be distinct")
    a, b = -dy / length, dx / length
    c = -(a * first[0] + b * first[1])
    return (a, b, c)


@dataclass(frozen=True)
class TableLocalizationResult:
    """Ordered inner-cushion geometry and its bed-plane mapping.

    Corners are clockwise in image coordinates: side ``i`` joins corner ``i``
    to corner ``(i + 1) % 4``.  The homography maps source points to a
    rectangular playable-bed coordinate system; rail strips have their own
    transform and must not be confused with this bed transform.
    """

    valid: bool
    image_size: tuple[int, int]
    bed_corners_xy: tuple[Point, ...]
    inner_cushion_lines: tuple[Line, ...]
    bed_homography: np.ndarray | None
    bed_homography_inverse: np.ndarray | None
    confidence: float = 0.0
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_corners(
        cls,
        corners: Sequence[Point],
        image_size: tuple[int, int],
        *,
        confidence: float = 1.0,
        warnings: Sequence[str] = (),
    ) -> "TableLocalizationResult":
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size values must be positive")
        if len(corners) != 4:
            raise ValueError("Exactly four table corners are required")
        ordered = tuple((float(x), float(y)) for x, y in corners)
        if abs(_signed_area(ordered)) <= 1e-6:
            raise ValueError("Table corners must enclose a non-zero area")
        if _signed_area(ordered) < 0:
            raise ValueError("Table corners must be clockwise in image coordinates")

        edge_lengths = [
            math.dist(ordered[index], ordered[(index + 1) % 4])
            for index in range(4)
        ]
        long_length = max(edge_lengths[0], edge_lengths[2])
        short_length = max(edge_lengths[1], edge_lengths[3])
        # Preserve aspect ratio while keeping strips at a useful pixel scale.
        bed_width = max(64, int(round(long_length)))
        bed_height = max(32, int(round(short_length)))
        source = np.asarray(ordered, dtype=np.float32)
        target = np.asarray(
            [
                [0.0, 0.0],
                [bed_width - 1.0, 0.0],
                [bed_width - 1.0, bed_height - 1.0],
                [0.0, bed_height - 1.0],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, target)
        inverse = cv2.getPerspectiveTransform(target, source)
        lines = tuple(
            _line_through(ordered[index], ordered[(index + 1) % 4])
            for index in range(4)
        )
        return cls(
            valid=True,
            image_size=(width, height),
            bed_corners_xy=ordered,
            inner_cushion_lines=lines,
            bed_homography=homography,
            bed_homography_inverse=inverse,
            confidence=float(confidence),
            warnings=tuple(warnings),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "image_size": list(self.image_size),
            "bed_corners_xy": [list(point) for point in self.bed_corners_xy],
            "inner_cushion_lines": [list(line) for line in self.inner_cushion_lines],
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }
