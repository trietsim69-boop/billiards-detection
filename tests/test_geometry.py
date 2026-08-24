from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from billiards.geometry import (  # noqa: E402
    PointObservation,
    RailFitConfig,
    fit_rails,
)


def points_on_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    fractions: list[float],
    noise_seed: int,
) -> list[PointObservation]:
    generator = np.random.default_rng(noise_seed)
    points: list[PointObservation] = []
    for fraction in fractions:
        x = start[0] + fraction * (end[0] - start[0])
        y = start[1] + fraction * (end[1] - start[1])
        noise = generator.normal(0.0, 0.6, size=2)
        points.append(PointObservation(x + float(noise[0]), y + float(noise[1])))
    return points


class RailGeometryTests(unittest.TestCase):
    def test_fits_trapezoid_with_outliers_and_vertical_safe_lines(self) -> None:
        corners = [(200.0, 100.0), (800.0, 150.0), (700.0, 500.0), (250.0, 550.0)]
        points = []
        points.extend(points_on_segment(corners[0], corners[1], [0.12, 0.25, 0.38, 0.62, 0.75, 0.88], 1))
        points.extend(points_on_segment(corners[1], corners[2], [0.25, 0.50, 0.75], 2))
        points.extend(points_on_segment(corners[2], corners[3], [0.12, 0.25, 0.38, 0.62, 0.75, 0.88], 3))
        points.extend(points_on_segment(corners[3], corners[0], [0.25, 0.50, 0.75], 4))
        points.extend([PointObservation(480.0, 310.0), PointObservation(900.0, 700.0)])
        np.random.default_rng(5).shuffle(points)

        result = fit_rails(
            points,
            image_size=(1000, 700),
            config=RailFitConfig(distance_threshold_ratio=0.004),
        )

        self.assertTrue(result.valid, result.warnings)
        self.assertEqual(len(result.rails), 4)
        self.assertEqual(sorted(len(rail.inlier_indices) for rail in result.rails), [3, 3, 6, 6])
        self.assertEqual(len(result.outlier_indices), 2)
        self.assertEqual(len(result.corners), 4)
        for expected in corners:
            nearest_error = min(
                np.linalg.norm(np.asarray(expected) - np.asarray(actual))
                for actual in result.corners
            )
            self.assertLess(nearest_error, 4.0)

    def test_fails_explicitly_when_there_are_too_few_points(self) -> None:
        result = fit_rails(
            [PointObservation(float(index), float(index)) for index in range(8)],
            image_size=(1920, 1080),
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.rails, ())
        self.assertIn("insufficient_dot_points:8<12", result.warnings)

    def test_rejects_non_positive_image_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            fit_rails([], image_size=(0, 1080))


if __name__ == "__main__":
    unittest.main()
