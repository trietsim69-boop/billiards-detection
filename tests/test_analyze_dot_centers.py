from __future__ import annotations

import unittest

from analyze_detector_errors import Detection
from analyze_dot_centers import (
    center_distance_model_px,
    inference_gain,
    match_centers,
)


def detection_at(x: float, y: float) -> Detection:
    return Detection(2, (x - 1.0, y - 1.0, x + 1.0, y + 1.0), 1.0)


class DotCenterMatchingTests(unittest.TestCase):
    def test_distance_is_reported_at_model_scale(self) -> None:
        left = detection_at(10.0, 10.0)
        right = detection_at(16.0, 18.0)

        self.assertAlmostEqual(center_distance_model_px(left, right, 0.5), 5.0)

    def test_matching_is_one_to_one(self) -> None:
        ground_truth = [detection_at(0.0, 0.0), detection_at(10.0, 0.0)]
        predictions = [detection_at(1.0, 0.0), detection_at(9.0, 0.0)]

        matches = match_centers(ground_truth, predictions, 2.0, gain=1.0)

        self.assertEqual(len(matches), 2)
        self.assertEqual({item.prediction_index for item in matches}, {0, 1})

    def test_augmenting_path_avoids_a_greedy_false_miss(self) -> None:
        ground_truth = [detection_at(0.0, 0.0), detection_at(4.0, 0.0)]
        predictions = [detection_at(1.0, 0.0), detection_at(-2.0, 0.0)]

        matches = match_centers(ground_truth, predictions, 3.0, gain=1.0)

        self.assertEqual(len(matches), 2)
        pairs = {
            (item.ground_truth_index, item.prediction_index) for item in matches
        }
        self.assertEqual(pairs, {(0, 1), (1, 0)})

    def test_inference_gain_uses_longest_image_side(self) -> None:
        self.assertAlmostEqual(inference_gain(1920, 1080, 640), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
