# Four-rail fitting report

- Input mode: `yolo`
- Images: `1`
- Structurally valid fits: `0/1`
- Held-out detector test split used: **no**
- Matched-view geometry evaluation set used: **no**

A structurally valid result means four supported lines produced a convex,
plausibly sized quadrilateral. It does not yet prove correct semantic rail
ordering or homography accuracy; inspect the overlays.

## Settings

- Rail distance threshold ratio: `0.004`
- Minimum inliers per rail: `3`
- Minimum total dots: `12`
- YOLO confidence: `0.05`

## Results

| Image | Dots | Rails | Supports | Area fraction | Valid | Warnings |
|---|---:|---:|---|---:|---|---|
| IMG_4529.jpeg | 11 | 0 |  |  | False | insufficient_dot_points:11<12 |

Open `contact_sheet.jpg` first, then inspect individual overlays and
JSON files for any suspicious result.
