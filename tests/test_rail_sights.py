from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from billiards.rail_regions import RailRegionConfig, extract_rail_regions  # noqa: E402
from billiards.rail_sights import RailSightConfig, detect_rail_sights  # noqa: E402
from billiards.table_localization import TableLocalizationResult  # noqa: E402


class RailSightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((300, 400, 3), (45, 95, 35), dtype=np.uint8)
        self.table = TableLocalizationResult.from_corners(
            [(70.0, 55.0), (330.0, 65.0), (350.0, 245.0), (45.0, 235.0)],
            (400, 300),
        )

    def test_white_rail_marks_are_found_and_bed_highlight_is_ignored(self) -> None:
        regions = extract_rail_regions(
            self.image,
            self.table,
            RailRegionConfig(strip_long_resolution=240, strip_depth_resolution=64),
        )
        expected_points: list[tuple[float, float]] = []
        for region in regions[:2]:
            for fraction in (0.2, 0.4, 0.6, 0.8):
                point = region.strip_to_source_point(
                    (fraction * (region.strip_image.shape[1] - 1), region.strip_image.shape[0] * 0.55)
                )
                expected_points.append(point)
                cv2.circle(self.image, tuple(round(value) for value in point), 4, (245, 245, 245), -1)
        cv2.circle(self.image, (200, 150), 10, (255, 255, 255), -1)

        result = detect_rail_sights(
            self.image,
            self.table,
            region_config=RailRegionConfig(strip_long_resolution=240, strip_depth_resolution=64),
            config=RailSightConfig(min_value=130, local_contrast=8.0, min_long_support=2, min_short_support=2),
        )

        self.assertGreaterEqual(len(result.candidates), 6)
        self.assertGreaterEqual(len(result.accepted), 4)
        for expected in expected_points[:4]:
            nearest = min(np.linalg.norm(np.asarray(expected) - np.asarray((item.x, item.y))) for item in result.candidates)
            self.assertLess(nearest, 8.0)
        self.assertTrue(all(np.linalg.norm(np.asarray((item.x, item.y)) - np.asarray((200.0, 150.0))) > 20.0 for item in result.candidates))

    def test_learned_candidate_outside_rails_is_not_accepted(self) -> None:
        result = detect_rail_sights(
            self.image,
            self.table,
            learned_candidates=[(10.0, 10.0, 0.99)],
            region_config=RailRegionConfig(strip_long_resolution=240, strip_depth_resolution=64),
        )
        self.assertFalse(any(abs(item.x - 10.0) < 1e-6 and abs(item.y - 10.0) < 1e-6 for item in result.candidates))


if __name__ == "__main__":
    unittest.main()
