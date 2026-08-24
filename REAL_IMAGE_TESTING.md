# Testing table-rail detection on real images

The rail-fitting command accepts either one image or a folder. New images do not
need labels: YOLO detects the rail dots, then the geometry module fits four lines
through their centres.

## First test image

Start with a deliberately easy image:

- the complete table and all four rails are inside the frame;
- most or all eighteen rail diamonds are visible;
- the table is reasonably large in the image;
- the image is sharp and not heavily compressed;
- players, score graphics, and glare do not hide the rails.

The current model was trained on broadcast images. A phone photograph is a useful
independent test, but failure on it may be domain shift rather than a rail-fitting
bug.

## Run one image

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\fit_table_rails.py `
  --source "C:\path\to\your\table_image.jpg" `
  --mode yolo `
  --output outputs\geometry\real-image-test
```

Quotes are important when the path contains spaces. The provisional Dot-candidate
confidence is `0.05`, read from `configs/geometry.yaml`. It is intentionally low
because downstream RANSAC rejects candidates that do not support a rail.

## Run a folder

```powershell
.\.venv\Scripts\python.exe scripts\fit_table_rails.py `
  --source "C:\path\to\your\images" `
  --mode yolo `
  --output outputs\geometry\real-folder-test
```

The command searches that folder recursively for JPG, JPEG, PNG, BMP, and WebP
images.

## Inspect the result

Open these files inside the selected output directory:

1. `contact_sheet.jpg` gives a quick visual overview.
2. `RAIL_FITTING_REPORT.md` lists Dot counts, rail support, validity, and warnings.
3. `<image-name>_rails.jpg` shows the detected Dot centres, four fitted lines, and
   the white table quadrilateral.
4. `<image-name>_rails.json` contains the exact points, normalized line equations,
   inlier indices, corners, thresholds, and warnings.

A good prototype result has:

- four colored lines following the physical diamond rows;
- a white quadrilateral whose corners are the four rail intersections;
- at least three supporting dots on every rail;
- no line through the table interior or background graphics;
- `valid=True` and no warnings in the overlay/report.

`valid=True` currently means structurally plausible. It does not yet prove that a
homography or projected ball position is accurate; those are later checkpoints.

## Interpreting common failures

| Warning | Meaning | First response |
|---|---|---|
| `insufficient_dot_points` | Fewer than twelve Dot candidates survived | Try a clearer/larger-table image; inspect YOLO Dot detections |
| `four_rails_not_found` | At least one rail lacked three collinear dots | Inspect which physical rail has missing dots or false positives |
| `quadrilateral_too_small` | The table occupies too little of the image for the current clear-image scope | Use a closer image or treat it as a later robustness case |
| `corner_outside_allowed_margin` | Fitted lines intersect implausibly far outside the image | Treat the automatic result as invalid |

For diagnosis only, a separate run may try a more conservative confidence without
overwriting the baseline:

```powershell
.\.venv\Scripts\python.exe scripts\fit_table_rails.py `
  --source "C:\path\to\your\table_image.jpg" `
  --mode yolo `
  --confidence 0.10 `
  --output outputs\geometry\real-image-test-conf010
```

Do not assume the higher threshold is better: validation rail success falls as
low-confidence but geometrically correct dots are removed. Keep both outputs and
compare their overlays.
