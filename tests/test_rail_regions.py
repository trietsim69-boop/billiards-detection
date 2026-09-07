from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from billiards.rail_regions import RailRegionConfig, extract_rail_regions  # noqa: E402
from billiards.table_localization import TableLocalizationResult  # noqa: E402


class RailRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((300, 400, 3), dtype=np.uint8)
        self.table = TableLocalizationResult.from_corners(
            [(70.0, 55.0), (330.0, 65.0), (350.0, 245.0), (45.0, 235.0)],
            (400, 300),
        )

    def test_extracts_four_traceable_regions(self) -> None:
        regions = extract_rail_regions(
            self.image,
            self.table,
            RailRegionConfig(strip_long_resolution=240, strip_depth_resolution=48),
        )

        self.assertEqual(len(regions), 4)
        self.assertEqual([region.side for region in regions], [0, 1, 2, 3])
        self.assertEqual([region.expected_sight_count for region in regions], [6, 3, 6, 3])
        for region in regions:
            source_point = region.source_polygon_xy[0]
            strip_point = region.source_to_strip_point(source_point)
            round_trip = region.strip_to_source_point(strip_point)
            self.assertLess(np.linalg.norm(np.asarray(source_point) - round_trip), 1e-3)
            self.assertGreater(region.valid_pixel_coverage, 0.8)

    def test_cropped_band_reports_invalid_coverage(self) -> None:
        cropped = self.image[:, :180]
        regions = extract_rail_regions(
            cropped,
            self.table,
            RailRegionConfig(strip_long_resolution=240, strip_depth_resolution=48),
        )
        self.assertTrue(any(region.warnings for region in regions))


if __name__ == "__main__":
    unittest.main()
