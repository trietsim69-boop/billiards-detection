"""Core billiards vision and geometry modules."""

from .geometry import PointObservation, RailFitConfig, RailFitResult, fit_rails
from .rail_regions import RailRegion, RailRegionConfig, extract_rail_regions
from .rail_sights import (
    RailSightCandidate,
    RailSightConfig,
    RailSightDetectionResult,
    detect_rail_sights,
)
from .table_localization import TableLocalizationResult

__all__ = [
    "PointObservation",
    "RailFitConfig",
    "RailFitResult",
    "fit_rails",
    "TableLocalizationResult",
    "RailRegion",
    "RailRegionConfig",
    "extract_rail_regions",
    "RailSightCandidate",
    "RailSightConfig",
    "RailSightDetectionResult",
    "detect_rail_sights",
]
