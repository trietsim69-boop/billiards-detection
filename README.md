# 8-ball pool table-state detection

Detect the balls and rail sights (diamonds) in a single broadcast image of an 8-ball table, fit the four rails, and map the table into normalized coordinates. Shot ranking is planned later.

**Status:** the ball and rail-dot detector is trained; rail fitting works. Homography, normalized table state and a manual four-corner fallback are next (`tasks/todo.md`).

## Best results so far

Everything below is on the 20-image validation split. The test split and the geometry holdout have not been used.

### Detector training

YOLO11n detecting 5 classes: Black, Cue, Dot (rail sight), Solid, Striped.

| Method | Image size | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Dot recall | Dot mAP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoke test (3 epochs) | 512 | 3 | 0.524 | 0.589 | 0.487 | 0.283 | – | – |
| Smoke test (3 epochs) | 640 | 3 | 0.499 | 0.718 | 0.650 | 0.437 | – | – |
| Baseline | 640 | 46 | 0.931 | 0.806 | 0.854 | 0.665 | 0.640 | 0.640 |
| **Best: `configs/train_yolo11n_960.yaml`** | **960** | **68** | **0.915** | **0.875** | **0.892** | **0.704** | **0.786** | **0.763** |
| Crop-aware fine-tune of the 960 model | 960 → 640 | 1 of 3 | 0.893 | 0.795 | 0.842 | 0.667 | – | – |

The best recipe starts from pretrained `yolo11n.pt`, uses batch 1 (nbs 2), early stopping with patience 20, `amp=False`, seed 42, and deterministic training. Mosaic, scale, translate and mixup are off, so the single-table scene stays intact; the only augmentations are horizontal flips and mild HSV jitter. Going from 640 to 960 px raised Dot recall from 0.640 to 0.786; every Dot box is smaller than 32×32 px.

The crop-aware fine-tune trained on 155 frames plus 620 overlapping tiles (AdamW, lr0 3e-4). Memory failures stopped it after 3 of 12 epochs and forced 640 px from epoch 2. It was not adopted.

### Dot centres

The geometry stage needs dot centres, not boxes. Each labelled dot is matched one-to-one with a prediction within 8 px at 960 scale; there are 359 labelled dots.

| Method | Matched | False positives | Missed | Recall | F1 | Valid rail fits |
|---|---:|---:|---:|---:|---:|---:|
| 960 model, full frame, conf 0.25 | 324 | 12 | 35 | 90.3% | 0.932 | 14/20 |
| + four overlapping 65% tiles, conf 0.50, Dot only | 341 | 14 | 18 | 95.0% | 0.955 | 16/20 |
| + tiles from the crop fine-tune (epoch 3) | 343 | 13 | 16 | 95.5% | 0.959 | 17/20 |
| Rail-first: rectified rail strips, conf 0.25 | 61 | 5 | 298 | 17.0% | – | – |

The 960 full-frame detector is the adopted default. Tiling adds 17 dots for four extra inference passes and was left experimental. Of the 16 dots still missed with fine-tuned tiles, 11 are in a single distant-table image.

### Rail fitting

`fit_rails` finds four collinear dot rows with consensus line fitting, intersects them, and rejects implausible quadrilaterals.

| Dot source | Four rails found | Structurally valid |
|---|---:|---:|
| Labelled dots | 20/20 | 19/20 (the miss is a distant table below the 3% area gate) |
| 640 model, conf 0.25 | 11/20 | 11/20 |
| 640 model, conf 0.05 | 20/20 | 19/20, all agreeing with label corners (median corner error 2.89 px) |

A low confidence works best for geometry: line fitting discards stray candidates, while a higher threshold drops real dots. The 0.05 threshold has not yet been re-swept for the 960 model.

## Setup

Developed on Windows with an NVIDIA MX550 (2 GB VRAM) in a project `.venv`: Python 3.12, PyTorch 2.9.0 + CUDA 12.6, torchvision 0.24.0, Ultralytics 8.4.124, OpenCV 5.0, PyYAML, matplotlib and pytest. Install the CUDA build of PyTorch from pytorch.org first, then `pip install ultralytics==8.4.124 pytest`.

Trained weights and run outputs are committed under `outputs/`; the best model is `outputs/detection/baseline-yolo11n-960/weights/best.pt`. With 2 GB VRAM, 960 px training needs batch 1, `workers=0` and `plots=false`; keep `amp=False`, because the AMP check fails on this GPU.

## Usage

Run everything from the repo root.

```powershell
# Train the best recipe. Pass an absolute project path: earlier runs with a
# relative one landed under runs\detect\outputs\detection\.
.\.venv\Scripts\yolo.exe detect train cfg=configs/train_yolo11n_960.yaml project="$PWD\outputs\detection" name=my-run

# Fit rails on your own image or folder (uses the 960 model by default)
.\.venv\Scripts\python.exe scripts\fit_table_rails.py --source "C:\path\to\image.jpg" --mode yolo --output outputs\geometry\my-test

# Evaluate on validation
.\.venv\Scripts\python.exe scripts\analyze_dot_centers.py      # dot-centre precision/recall sweep
.\.venv\Scripts\python.exe scripts\analyze_detector_errors.py  # per-class errors and gallery
.\.venv\Scripts\python.exe scripts\sweep_rail_confidence.py    # choose the Dot confidence for rail fitting

# Tests
.\.venv\Scripts\pytest.exe -q
```

`REAL_IMAGE_TESTING.md` explains how to read the rail-fitting overlays and warnings.

## Data

[Pix2Pockets 8-Ball Pool v3](https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o/dataset/3) (CC BY 4.0). Train only on `data/processed/pix2pockets_v3/data.yaml`, which splits the images into 155 train, 20 validation and 20 test. Another 52 matched-view images (25 situations) are reserved for geometry evaluation.

## Repository

```text
configs/train_yolo11n_960.yaml   best training recipe
configs/geometry.yaml            rail-fitting thresholds and Dot confidence
src/billiards/geometry.py        four-rail fitting from dot centres
scripts/                         evaluation and rail-fitting CLIs
tests/
outputs/detection/baseline-yolo11n-960/   best weights, per-epoch results.csv, training args
TABLE_DETECTION_PROGRESS.md      full experiment log and decisions
```

The rail-first, crop-tiling and crop fine-tuning code was removed after those experiments. It can be recovered from commit `d6eb42e`, e.g. `git show d6eb42e:src/billiards/crop_detection.py`.
