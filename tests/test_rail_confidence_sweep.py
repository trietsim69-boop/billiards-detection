from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from sweep_rail_confidence import corner_alignment_error  # noqa: E402


class RailConfidenceSweepTests(unittest.TestCase):
    def test_corner_error_is_independent_of_corner_order(self) -> None:
        reference = [(10.0, 10.0), (90.0, 10.0), (90.0, 50.0), (10.0, 50.0)]
        predicted = [reference[2], reference[0], reference[3], reference[1]]

        error = corner_alignment_error(predicted, reference)

        self.assertIsNotNone(error)
        assert error is not None
        self.assertEqual(error, (0.0, 0.0))

    def test_corner_error_requires_four_corners(self) -> None:
        self.assertIsNone(corner_alignment_error([(0.0, 0.0)], []))


if __name__ == "__main__":
    unittest.main()
