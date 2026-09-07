from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from billiards.crop_detection import BoxDetection, CropBox, crop_labels, overlapping_tiles, supplement_detections


class CropDetectionTests(unittest.TestCase):
    def test_tiles_cover_every_pixel_without_changing_aspect(self):
        for width, height in ((1920,1080), (101,73)):
            covered = np.zeros((height,width), dtype=bool)
            for tile in overlapping_tiles((width,height)):
                covered[tile.y1:tile.y2,tile.x1:tile.x2] = True
                self.assertLess(abs(tile.width/tile.height-width/height), 0.03)
            self.assertTrue(covered.all())

    def test_crop_labels_clip_partial_objects_and_preserve_class(self):
        labels = [(2,0.5,0.5,0.2,0.2), (4,0.95,0.5,0.08,0.1)]
        actual = crop_labels(labels, (100,100), CropBox(50,25,100,75))
        self.assertEqual([p[0] for p in actual], [2,4])
        np.testing.assert_allclose(actual[0][1:], [0.1,0.5,0.2,0.4])
        self.assertTrue(all(0 <= v <= 1 for p in actual for v in p[1:]))

    def test_merging_preserves_baseline_and_nearby_distinct_dots(self):
        primary = BoxDetection(2, (10,10,14,14), .25)
        duplicate = BoxDetection(2, (10.5,10.5,14.5,14.5), .9, "tile")
        other = BoxDetection(2, (18,10,22,14), .8, "tile")
        self.assertEqual(supplement_detections([primary], [duplicate,other]), (primary,other))

    def test_tile_coordinates_translate_exactly(self):
        point = BoxDetection(2,(1,2,5,6),.8).translate(100,50,"tile0")
        self.assertEqual(point.box_xyxy,(101,52,105,56))
        self.assertEqual(point.center,(103,54))


if __name__ == "__main__":
    unittest.main()
