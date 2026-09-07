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
        if not np.isfinite(ordered).all():
            raise ValueError("Table corners must be finite")
        if not cv2.isContourConvex(np.asarray(ordered, dtype=np.float32)):
            raise ValueError("Table corners must form a convex quadrilateral")
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


@dataclass(frozen=True)
class TableLocalizationConfig:
    """Provisional image-only proposal settings, not a calibrated bed oracle."""

    max_resolution: int = 640
    min_saturation: int = 70
    min_value: int = 45
    hue_radius: int = 10
    min_area_fraction: float = 0.01
    max_area_fraction: float = 0.90
    min_mask_iou: float = 0.80


def localize_table(
    image: np.ndarray, config: TableLocalizationConfig | None = None
) -> TableLocalizationResult:
    """Propose a cloth quadrilateral from dominant saturated image colours.

    This initial locator is for detection experiments. Colour/shape agreement
    does not certify the cushion nose or metric bed mapping. No sight labels or
    YOLO predictions enter this function. Cropped proposals are rejected.
    """
    config = config or TableLocalizationConfig()
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty BGR image")
    height, width = image.shape[:2]
    scale = min(1.0, config.max_resolution / max(height, width))
    small = cv2.resize(image, (round(width * scale), round(height * scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    eligible = (hsv[:, :, 1] >= config.min_saturation) & (hsv[:, :, 2] >= config.min_value)
    histogram = np.bincount(hsv[:, :, 0][eligible], minlength=180)
    peaks: list[int] = []
    for hue in np.argsort(-histogram, kind="stable"):
        if histogram[hue] == 0:
            break
        if all(min(abs(int(hue) - peak), 180 - abs(int(hue) - peak)) > config.hue_radius * 2 for peak in peaks):
            peaks.append(int(hue))
        if len(peaks) == 4:
            break
    best = None
    for hue in peaks:
        distance = np.abs(hsv[:, :, 0].astype(float) - hue)
        mask = (eligible & (np.minimum(distance, 180 - distance) <= config.hue_radius)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(contour)
            fraction = area / mask.size
            if not config.min_area_fraction <= fraction <= config.max_area_fraction:
                continue
            hull = cv2.convexHull(contour)
            for epsilon in (0.01, 0.02, 0.03, 0.04):
                quad = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
                if len(quad) != 4 or not cv2.isContourConvex(quad):
                    continue
                points = quad[:, 0, :].astype(float)
                if (points[:, 0] <= 1).any() or (points[:, 0] >= small.shape[1] - 2).any() or (points[:, 1] <= 1).any() or (points[:, 1] >= small.shape[0] - 2).any():
                    continue
                filled = np.zeros_like(mask)
                polygon = np.zeros_like(mask)
                cv2.drawContours(filled, [contour], -1, 255, -1)
                cv2.fillConvexPoly(polygon, quad, 255)
                iou = np.count_nonzero(filled & polygon) / max(1, np.count_nonzero(filled | polygon))
                if iou < config.min_mask_iou:
                    continue
                center = points.mean(axis=0)
                points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
                points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
                # Existing rail module assumes sides 0/2 are long. This is an
                # image-length heuristic only; save that limitation explicitly.
                lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
                if lengths[1] + lengths[3] > lengths[0] + lengths[2]:
                    points = np.roll(points, -1, axis=0)
                score = area * iou
                if best is None or score > best[0]:
                    points *= np.asarray([width / small.shape[1], height / small.shape[0]])
                    best = (score, points, iou)
    if best is None:
        return TableLocalizationResult(False, (width, height), (), (), None, None, warnings=("cloth_quad_not_found",))
    return TableLocalizationResult.from_corners(
        best[1], (width, height), confidence=best[2],
        warnings=("experimental_cloth_boundary_not_reviewed", "long_side_inferred_from_image_length"),
    )
