"""Classical and learned white rail-sight candidates with lattice filtering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from .geometry import PointObservation
from .rail_regions import RailRegion, RailRegionConfig, extract_rail_regions
from .table_localization import TableLocalizationResult


@dataclass(frozen=True)
class RailSightConfig:
    min_value: int = 145
    max_saturation: int = 105
    local_contrast: float = 14.0
    blur_kernel: int = 15
    min_component_area: int = 3
    max_component_area: int = 1800
    min_solidity: float = 0.35
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 4.0
    sight_row_min: float = 0.08
    sight_row_max: float = 0.92
    lattice_tolerance: float = 0.095
    lattice_residual: float = 0.075
    min_long_support: int = 3
    min_short_support: int = 2
    dedup_distance_ratio: float = 0.025
    classical_weight: float = 0.75


@dataclass(frozen=True)
class RailSightCandidate:
    x: float
    y: float
    confidence: float
    source: str
    rail_side: int | None = None
    strip_u: float | None = None
    strip_v: float | None = None
    area: float | None = None
    solidity: float | None = None
    local_contrast: float | None = None
    reason: str | None = None

    def as_point(self) -> PointObservation:
        return PointObservation(self.x, self.y, self.confidence)

    def as_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "source": self.source,
            "rail_side": self.rail_side,
            "strip_u": self.strip_u,
            "strip_v": self.strip_v,
            "area": self.area,
            "solidity": self.solidity,
            "local_contrast": self.local_contrast,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RailSightDetectionResult:
    candidates: tuple[RailSightCandidate, ...]
    accepted: tuple[RailSightCandidate, ...]
    rejected: tuple[RailSightCandidate, ...]
    per_rail: tuple[tuple[RailSightCandidate, ...], ...]
    warnings: tuple[str, ...]

    @property
    def accepted_centres(self) -> tuple[PointObservation, ...]:
        return tuple(candidate.as_point() for candidate in self.accepted)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_count": len(self.candidates),
            "accepted_count": len(self.accepted),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "accepted": [candidate.as_dict() for candidate in self.accepted],
            "rejected": [candidate.as_dict() for candidate in self.rejected],
            "warnings": list(self.warnings),
        }


def _valid_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _component_solidity(contour: np.ndarray, area: float) -> float:
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    return area / hull_area if hull_area > 1e-9 else 0.0


def _classical_candidates(
    region: RailRegion, config: RailSightConfig
) -> list[RailSightCandidate]:
    strip = region.strip_image
    if strip.ndim == 2:
        gray = strip
        saturation = np.zeros_like(gray)
    else:
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        gray = hsv[:, :, 2]
    kernel = _valid_kernel(config.blur_kernel)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    local = gray.astype(np.float32) - background.astype(np.float32)
    mask = (
        (gray >= config.min_value)
        & (saturation <= config.max_saturation)
        & (local >= config.local_contrast)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[RailSightCandidate] = []
    strip_height, strip_width = strip.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not config.min_component_area <= area <= config.max_component_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect = width / height
        solidity = _component_solidity(contour, area)
        if not config.min_aspect_ratio <= aspect <= config.max_aspect_ratio:
            continue
        if solidity < config.min_solidity:
            continue
        center = (x + width / 2.0, y + height / 2.0)
        v = center[1] / max(1.0, strip_height - 1.0)
        if not config.sight_row_min <= v <= config.sight_row_max:
            continue
        source_x, source_y = region.strip_to_source_point(center)
        mean_contrast = float(np.mean(local[y : y + height, x : x + width]))
        score = min(1.0, max(0.0, mean_contrast / 80.0))
        candidates.append(
            RailSightCandidate(
                x=source_x,
                y=source_y,
                confidence=score * config.classical_weight,
                source="classical",
                rail_side=region.side,
                strip_u=center[0] / max(1.0, strip_width - 1.0),
                strip_v=v,
                area=area,
                solidity=solidity,
                local_contrast=mean_contrast,
            )
        )
    return candidates


def _as_learned_candidate(item: object) -> RailSightCandidate:
    if isinstance(item, PointObservation):
        return RailSightCandidate(item.x, item.y, item.confidence, "learned")
    if isinstance(item, RailSightCandidate):
        return item
    if isinstance(item, Sequence) and len(item) >= 2:
        confidence = float(item[2]) if len(item) >= 3 else 1.0
        return RailSightCandidate(float(item[0]), float(item[1]), confidence, "learned")
    raise TypeError("learned_candidates must contain points or PointObservations")


def _assign_to_region(
    candidate: RailSightCandidate,
    regions: Sequence[RailRegion],
) -> RailSightCandidate | None:
    possible: list[tuple[float, RailRegion, tuple[float, float]]] = []
    for region in regions:
        u, v = region.source_to_strip_point((candidate.x, candidate.y))
        width, height = region.strip_image.shape[1], region.strip_image.shape[0]
        if -1.0 <= u <= width and -1.0 <= v <= height:
            # Prefer candidates well inside a band when corner regions overlap.
            edge_distance = min(u, v, width - u, height - v)
            possible.append((edge_distance, region, (u, v)))
    if not possible:
        return None
    _, region, (u, v) = max(possible, key=lambda item: item[0])
    width, height = region.strip_image.shape[1], region.strip_image.shape[0]
    return RailSightCandidate(
        x=candidate.x,
        y=candidate.y,
        confidence=candidate.confidence,
        source=candidate.source,
        rail_side=region.side,
        strip_u=max(0.0, min(1.0, u / max(1.0, width - 1.0))),
        strip_v=max(0.0, min(1.0, v / max(1.0, height - 1.0))),
        area=candidate.area,
        solidity=candidate.solidity,
        local_contrast=candidate.local_contrast,
    )


def _deduplicate(candidates: Sequence[RailSightCandidate], image_size: tuple[int, int], config: RailSightConfig) -> list[RailSightCandidate]:
    threshold = config.dedup_distance_ratio * max(image_size)
    kept: list[RailSightCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = next(
            (
                existing
                for existing in kept
                if math.hypot(candidate.x - existing.x, candidate.y - existing.y) <= threshold
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        elif duplicate.source != candidate.source:
            # Keep the stronger candidate but preserve the union provenance.
            index = kept.index(duplicate)
            kept[index] = RailSightCandidate(
                x=duplicate.x,
                y=duplicate.y,
                confidence=max(duplicate.confidence, candidate.confidence),
                source="union",
                rail_side=duplicate.rail_side,
                strip_u=duplicate.strip_u,
                strip_v=duplicate.strip_v,
                area=duplicate.area,
                solidity=duplicate.solidity,
                local_contrast=duplicate.local_contrast,
            )
    return kept


def _lattice_filter(
    candidates: Sequence[RailSightCandidate],
    region: RailRegion,
    config: RailSightConfig,
) -> tuple[list[RailSightCandidate], list[RailSightCandidate], str | None]:
    ordered = sorted(candidates, key=lambda item: float(item.strip_u or 0.0))
    expected_count = region.expected_sight_count
    expected = np.linspace(1.0 / (expected_count + 1), expected_count / (expected_count + 1), expected_count)
    if region.warnings:
        return [], [candidate for candidate in ordered], f"rail_{region.side}_insufficient_coverage"
    if not ordered:
        return [], [], f"rail_{region.side}_no_candidates"

    assigned: list[tuple[float, RailSightCandidate]] = []
    used: set[int] = set()
    for position in expected:
        choices = [
            (abs(float(candidate.strip_u or 0.0) - position), index, candidate)
            for index, candidate in enumerate(ordered)
            if index not in used
        ]
        if not choices:
            continue
        distance, index, candidate = min(choices, key=lambda item: (item[0], item[1]))
        if distance <= config.lattice_tolerance:
            assigned.append((distance, candidate))
            used.add(index)

    minimum = config.min_long_support if region.long_side else config.min_short_support
    residual = float(np.mean([distance for distance, _ in assigned])) if assigned else math.inf
    if len(assigned) < minimum or residual > config.lattice_residual:
        return [], [candidate for candidate in ordered], f"rail_{region.side}_weak_lattice"
    accepted = [candidate for _, candidate in assigned]
    rejected = [candidate for index, candidate in enumerate(ordered) if index not in used]
    return accepted, rejected, None


def detect_rail_sights(
    image: np.ndarray,
    table: TableLocalizationResult,
    learned_candidates: Iterable[object] = (),
    config: RailSightConfig | None = None,
    region_config: RailRegionConfig | None = None,
) -> RailSightDetectionResult:
    """Detect rail sights without loading a YOLO model.

    ``learned_candidates`` are injected by callers as source-image centres;
    this module owns only region assignment, classical proposals, deduplication,
    and the rail-wise spacing decision.
    """
    config = config or RailSightConfig()
    regions = extract_rail_regions(image, table, region_config)
    raw: list[RailSightCandidate] = []
    for region in regions:
        raw.extend(_classical_candidates(region, config))
    for item in learned_candidates:
        assigned = _assign_to_region(_as_learned_candidate(item), regions)
        if assigned is not None:
            raw.append(assigned)
    assigned_candidates = [
        assigned
        for candidate in raw
        if (assigned := _assign_to_region(candidate, regions)) is not None
    ]
    candidates = _deduplicate(assigned_candidates, table.image_size, config)
    per_rail: list[tuple[RailSightCandidate, ...]] = []
    accepted: list[RailSightCandidate] = []
    rejected: list[RailSightCandidate] = []
    warnings: list[str] = list(table.warnings)
    for region in regions:
        warnings.extend(f"rail_{region.side}_{warning}" for warning in region.warnings)
    for region in regions:
        rail_candidates = [candidate for candidate in candidates if candidate.rail_side == region.side]
        rail_accepted, rail_rejected, warning = _lattice_filter(rail_candidates, region, config)
        per_rail.append(tuple(rail_accepted))
        accepted.extend(rail_accepted)
        rejected.extend(rail_rejected)
        if warning is not None:
            warnings.append(warning)
    return RailSightDetectionResult(
        candidates=tuple(candidates),
        accepted=tuple(sorted(accepted, key=lambda item: (item.rail_side or 0, item.strip_u or 0.0))),
        rejected=tuple(rejected),
        per_rail=tuple(per_rail),
        warnings=tuple(warnings),
    )
