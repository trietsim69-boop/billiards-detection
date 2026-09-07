from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from billiards.table_localization import TableLocalizationResult, localize_table


class TableLocalizationTests(unittest.TestCase):
    def test_recovers_coloured_cloth_with_ball_holes(self):
        for colour in ((160, 100, 30), (30, 140, 45)):
            image = np.full((300, 500, 3), 30, np.uint8)
            corners = np.asarray([(80,60), (400,75), (430,240), (50,230)], np.int32)
            cv2.fillConvexPoly(image, corners, colour)
            cv2.circle(image, (230,150), 10, (255,255,255), -1)
            result = localize_table(image)
            self.assertTrue(result.valid)
            for point in corners:
                self.assertLess(min(np.linalg.norm(point - np.asarray(p)) for p in result.bed_corners_xy), 3)

    def test_blank_or_cropped_images_fail(self):
        for image in (np.zeros((100,200,3), np.uint8), np.full((100,200,3), (30,140,45), np.uint8)):
            self.assertFalse(localize_table(image).valid)

    def test_invalid_corners_rejected(self):
        for points in ([(0,0),(100,0),(20,20),(0,100)], [(0,0),(100,0),(float('nan'),100),(0,100)]):
            with self.assertRaises(ValueError):
                TableLocalizationResult.from_corners(points, (200,200))


if __name__ == "__main__":
    unittest.main()
