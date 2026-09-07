"""Aspect-preserving tile geometry and conservative detector-box merging.

Tiles use only image dimensions, so table localization cannot exclude sights.
No resampling happens here; YOLO applies its normal letterbox resize. Full-frame
predictions retain priority when tile predictions describe the same object.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class BoxDetection:
    class_id: int
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    source: str = "full"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def translate(self, x: float, y: float, source: str) -> "BoxDetection":
        x1, y1, x2, y2 = self.box_xyxy
        return BoxDetection(self.class_id, (x1+x, y1+y, x2+x, y2+y), self.confidence, source)


def overlapping_tiles(image_size: tuple[int, int], fraction: float = 0.65) -> tuple[CropBox, ...]:
    """Four corner tiles, each spanning the same fraction of width and height."""
    width, height = image_size
    if width < 2 or height < 2 or not 0.5 <= fraction < 1:
        raise ValueError("Positive image dimensions >=2 and fraction in [0.5, 1) required")
    crop_width, crop_height = math.ceil(width * fraction), math.ceil(height * fraction)
    return tuple(dict.fromkeys(
        CropBox(x, y, x + crop_width, y + crop_height)
        for y in (0, height - crop_height) for x in (0, width - crop_width)
    ))


def crop_labels(
    labels: Sequence[tuple[int, float, float, float, float]],
    image_size: tuple[int, int], crop: CropBox,
) -> list[tuple[int, float, float, float, float]]:
    """Clip normalized YOLO boxes into a tile, retaining all visible classes.

    Fragments smaller than one original-image pixel in either dimension are
    omitted. Other partial objects are labelled rather than taught as background.
    """
    width, height = image_size
    result = []
    for cls, cx, cy, bw, bh in labels:
        x1 = max(crop.x1, (cx - bw/2) * width)
        y1 = max(crop.y1, (cy - bh/2) * height)
        x2 = min(crop.x2, (cx + bw/2) * width)
        y2 = min(crop.y2, (cy + bh/2) * height)
        if x2-x1 < 1 or y2-y1 < 1:
            continue
        result.append((int(cls), ((x1+x2)/2-crop.x1)/crop.width,
                       ((y1+y2)/2-crop.y1)/crop.height,
                       (x2-x1)/crop.width, (y2-y1)/crop.height))
    return result


def duplicate_boxes(left: BoxDetection, right: BoxDetection) -> bool:
    if left.class_id != right.class_id:
        return False
    a, b = left.box_xyxy, right.box_xyxy
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    aa, ab = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    iou = intersection / max(1e-9, aa+ab-intersection)
    diagonal = min(math.hypot(a[2]-a[0], a[3]-a[1]), math.hypot(b[2]-b[0], b[3]-b[1]))
    return iou >= 0.35 or math.dist(left.center, right.center) <= 0.5 * diagonal


def supplement_detections(
    primary: Sequence[BoxDetection], extra: Sequence[BoxDetection],
) -> tuple[BoxDetection, ...]:
    """Preserve every primary result and append nonduplicate extras, score first."""
    kept = list(primary)
    for candidate in sorted(extra, key=lambda p: p.confidence, reverse=True):
        if not any(duplicate_boxes(candidate, existing) for existing in kept):
            kept.append(candidate)
    return tuple(kept)
