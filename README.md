# 8-ball pool ball and rail-sight detection

A YOLO11 detector for broadcast images of 8-ball pool. It finds the balls (Black, Cue, Solid, Striped) and the rail sights, or diamonds ("Dot"), then fits the four table rails through the detected sights.

The best model is **YOLO11s at 1920 px**: validation mAP50 0.952, Dot recall 0.844, and 19/20 validation tables fitted correctly. Weights: `outputs/yolo11s-1920/weights/best.pt`.

## Results

All numbers are on the 20-image validation split; the 20-image test split was never used.

| Model | Image size | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Dot recall | Dot mAP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n smoke test (3 epochs) | 512 | 3 | 0.524 | 0.589 | 0.487 | 0.283 | – | – |
| YOLO11n smoke test (3 epochs) | 640 | 3 | 0.499 | 0.718 | 0.650 | 0.437 | – | – |
| YOLO11n baseline | 640 | 46 | 0.931 | 0.806 | 0.854 | 0.665 | 0.640 | 0.640 |
| [YOLO11n, laptop](outputs/yolo11n-960/RESULTS.md) | 960 | 68 | 0.915 | 0.875 | 0.892 | 0.704 | 0.786 | 0.763 |
| [YOLO11n, Colab](outputs/yolo11n-1280/RESULTS.md) | 1280 | 55 | 0.913 | 0.873 | 0.926 | 0.734 | 0.778 | 0.839 |
| **[YOLO11s, Colab](outputs/yolo11s-1920/RESULTS.md)** | **1920** | **27** | **0.938** | **0.929** | **0.952** | **0.770** | **0.844** | **0.895** |

Each linked model has its own `RESULTS.md` with per-class tables, settings and evaluations. Every run kept the single-table scene intact: mosaic, scale, translate and mixup off, horizontal flips and mild HSV jitter only. Resolution was the biggest lever, because every Dot box is under 32×32 px.

**Dot centres.** Rail fitting needs dot centres, not boxes, so each model is also scored by matching labelled dots to predictions within the same physical distance (8 px at 960 = 16 px at 1920). At confidence 0.25 the 960 model finds 324 of 359 dots (12 false positives), while the 1920 model finds 354 (86 false positives) and 358 at confidence 0.05. The extra false positives cost little, because line fitting rejects candidates that are not collinear.

**Rail fitting.** With the 1920 model, every Dot confidence from 0.05 to 0.20 fits 19 of 20 validation tables in agreement with the labelled corners (median corner error ≈ 1.9 px). That matches labelled dots, which also give 19/20: the remaining table is too distant for the 3% area gate. The repo uses confidence 0.15.

**Tried and dropped:**
- Tiling the 960 model's input into four 65% crops found 341/359 dots instead of 324. It costs four extra passes and was superseded by the 1920 model.
- Fine-tuning on those crops (3 of 12 epochs before the laptop ran out of memory) added only 2 more.
- Detecting on rectified rail strips found 61/359.

## Usage

Run from the repo root with the project virtual environment. Every script defaults to the 1920 model.

```powershell
# Fit rails on your own image or folder
.\.venv\Scripts\python.exe scripts\fit_table_rails.py --source "C:\path\to\image.jpg" --output outputs\yolo11s-1920\my-test

# Re-run the validation evaluations
.\.venv\Scripts\python.exe scripts\analyze_dot_centers.py      # dot-centre precision/recall sweep
.\.venv\Scripts\python.exe scripts\analyze_detector_errors.py  # per-class errors and gallery
.\.venv\Scripts\python.exe scripts\sweep_rail_confidence.py    # Dot confidence vs rail-fitting success

# Tests
.\.venv\Scripts\pytest.exe -q
```

Pass `--weights outputs/yolo11n-960/weights/best.pt --imgsz 960` to use the smaller model, or `--device cpu` without a CUDA GPU.

Rail fitting writes an overlay (`*_rails.jpg`), a JSON file and `RAIL_FITTING_REPORT.md`. A good result has four lines along the diamond rows, a quadrilateral at their intersections, at least three dots per rail, and `valid=True`. Warnings:

| Warning | Meaning |
|---|---|
| `insufficient_dot_points` | fewer than 12 Dot candidates survived |
| `four_rails_not_found` | a rail lacked three collinear dots |
| `quadrilateral_too_small` | the table covers under 3% of the image |
| `corner_outside_allowed_margin` | the rails intersect implausibly far outside the image |

`valid=True` means structurally plausible; it does not prove the corners are exact. The model was trained on broadcast frames, so phone photos are a domain shift: one test photo with the 960 model produced too few dots.

## Training

The best recipe is [`configs/train_yolo11s_1920.yaml`](configs/train_yolo11s_1920.yaml). It needs a ~16 GB GPU and was trained on a free Google Colab T4. Use a T4 runtime, store a read-only GitHub token for this repo as the Colab secret `GITHUB_TOKEN`, and run:

```python
from google.colab import drive, userdata
drive.mount('/content/drive')
token = userdata.get('GITHUB_TOKEN')
!git clone -q https://{token}@github.com/trietsim69-boop/8ballpool.git /content/8ballpool
%cd /content/8ballpool
!pip install -q ultralytics==8.4.124
!yolo detect train cfg=configs/train_yolo11s_1920.yaml project=/content/drive/MyDrive/8ballpool/outputs
```

- **Batch size:** keep `nbs` equal to `batch`; a smaller `nbs` silently increases weight decay. On out-of-memory errors use `batch=2 nbs=2`.
- **Disconnects:** results save to Google Drive and survive them. Resume with `!yolo train resume model=<run>/weights/last.pt`.
- **Other runs:** the 960 and 1280 runs' exact settings are in `outputs/*/args.yaml`.

Local development used Windows, Python 3.12, PyTorch 2.9 + CUDA 12.6, Ultralytics 8.4.124 and OpenCV 5 on an NVIDIA MX550 (2 GB). That GPU can train at most YOLO11n at 960 px with batch 1 and `amp=False`.

## Data

[8-Ball Pool v3](https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o/dataset/3) from Roboflow (CC BY 4.0), split in `data/processed/pix2pockets_v3/`:

| Split | Images |
|---|---:|
| train | 155 |
| valid | 20 |
| test | 20 |
| geometry_eval (matched views of 25 situations) | 52 |

Near-duplicates are kept within one split. Train only with `data/processed/pix2pockets_v3/data.yaml`.

## Repository

```text
configs/train_yolo11s_1920.yaml   best training recipe
configs/geometry.yaml             rail-fitting thresholds and Dot confidence
scripts/geometry.py               four-rail fitting from dot centres
scripts/*.py                      evaluation and rail-fitting command-line tools
tests/                            unit tests
outputs/<model>/RESULTS.md        results per trained model, with weights and evaluations
data/processed/pix2pockets_v3/    dataset
```
