# YOLO11s at 1920 px — best model

The default model for every script in this repo: `weights/best.pt` (19.4 MB, 9.4M parameters). Trained on a free Google Colab Tesla T4 (15 GB).

## Training

[`configs/train_yolo11s_1920.yaml`](../../configs/train_yolo11s_1920.yaml): pretrained `yolo11s.pt`, imgsz 1920 (the frames' native width, so dots are not downscaled), batch 4 / nbs 4, `cache=ram`, `amp=True`, seed 42, deterministic. Mosaic, scale, translate and mixup off; horizontal flips and mild HSV jitter only. 47 epochs in 0.42 hours; early stopping kept epoch 27. Exact settings: [`args.yaml`](args.yaml); per-epoch metrics: [`results.csv`](results.csv), [`results.png`](results.png).

## Validation (20 images, Ultralytics `val` of `best.pt`)

| Class | Precision | Recall | mAP50 | mAP50-95 | Δ mAP50 vs 960 |
|---|---:|---:|---:|---:|---:|
| All | 0.938 | 0.929 | 0.952 | 0.770 | +0.060 |
| Black | 0.954 | 0.900 | 0.943 | 0.807 | +0.053 |
| Cue | 0.960 | 1.000 | 0.995 | 0.905 | +0.005 |
| Dot | 0.854 | 0.844 | 0.895 | 0.494 | +0.132 |
| Solid | 0.960 | 0.948 | 0.963 | 0.829 | +0.075 |
| Striped | 0.965 | 0.952 | 0.966 | 0.815 | +0.039 |

Every class improves on the 960 model, and this is the first run to raise Dot recall. Model size and resolution changed together, so their separate effects are unknown. Inference takes 37 ms per image on a T4. Curves and confusion matrices: `Box*_curve.png`, `confusion_matrix*.png`. Predictions on eight validation frames: [`val_batch0_pred.jpg`](val_batch0_pred.jpg), with labels in [`val_batch0_labels.jpg`](val_batch0_labels.jpg). End-to-end examples (balls at confidence ≥ 0.5, sights at 0.15, fitted rails) on three validation frames are in [`examples/`](examples/).

## Dot centres

359 labelled dots, one-to-one matching within 16 px at 1920 scale (the same physical distance as 8 px at 960). Details and a 20-image gallery: [`dot-centers/`](dot-centers/DOT_CENTER_REPORT.md).

| Confidence | Found | False positives | Missed | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 358 | 292 | 1 | 0.551 | 0.997 | 0.710 |
| 0.20 | 356 | 112 | 3 | 0.761 | 0.992 | 0.861 |
| 0.25 | 354 | 86 | 5 | 0.805 | 0.986 | 0.886 |
| 0.30 | 348 | 51 | 11 | 0.872 | 0.969 | 0.918 |
| 0.40 | 320 | 14 | 39 | 0.958 | 0.891 | **0.924** |
| 0.50 | 262 | 4 | 97 | 0.985 | 0.730 | 0.838 |

The model finds almost every dot (358/359 at 0.05; the 960 model's best was 332) but is less confident, so its best single F1 is slightly below the 960 model's 0.933. Rail fitting tolerates the extra false positives, which makes recall the number that matters.

## Rail fitting

`scripts/sweep_rail_confidence.py` on the 20 validation images; agreement means fitted corners within 1% of the image diagonal (mean) and 2% (max) of the corners fitted from labelled dots. Details: [`rail-sweep/`](rail-sweep/RAIL_CONFIDENCE_SWEEP.md).

| Confidence | Mean dot candidates | Four rails | Valid and agreeing | Median corner error |
|---:|---:|---:|---:|---:|
| 0.05 | 32.5 | 20/20 | 19/20 | 2.03 px |
| 0.10 | 28.6 | 20/20 | 19/20 | 1.87 px |
| **0.15** | 25.8 | 20/20 | **19/20** | 1.89 px |
| 0.20 | 23.4 | 20/20 | 19/20 | 1.74 px |
| 0.25 | 22.0 | 19/20 | 18/20 | 1.57 px |
| 0.30 | 19.9 | 18/20 | 18/20 | 1.54 px |
| 0.40 | 16.7 | 14/20 | 14/20 | 1.34 px |
| 0.50 | 13.3 | 9/20 | 9/20 | 0.67 px |

19/20 is the ceiling: labelled dots also give 19/20, because one distant table falls below the 3% image-area gate. The repo uses confidence 0.15 (`configs/geometry.yaml`), mid-way through the 0.05–0.20 plateau; overlays for that setting are in `rail-sweep/conf_015/`.
