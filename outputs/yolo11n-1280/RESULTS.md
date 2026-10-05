# YOLO11n at 1280 px — Colab

The 960 recipe at a higher resolution, trained on a free Google Colab Tesla T4 (15 GB). Superseded by [YOLO11s at 1920](../yolo11s-1920/RESULTS.md).

## Training

Same augmentation policy and seed as the [960 run](../yolo11n-960/RESULTS.md), with `imgsz=1280 batch=8 nbs=8 workers=2 amp=True plots=True`. Ultralytics 8.4.124, torch 2.11 + CUDA 13. 75 epochs in 0.25 hours; early stopping kept epoch 55. Exact settings: [`args.yaml`](args.yaml); per-epoch metrics: [`results.csv`](results.csv), [`results.png`](results.png).

## Validation (20 images, Ultralytics `val` of `best.pt`)

| Class | Precision | Recall | mAP50 | mAP50-95 | Δ mAP50 vs 960 |
|---|---:|---:|---:|---:|---:|
| All | 0.913 | 0.873 | 0.926 | 0.734 | +0.034 |
| Black | 0.962 | 0.850 | 0.976 | 0.805 | +0.086 |
| Cue | 0.857 | 1.000 | 0.993 | 0.886 | +0.003 |
| Dot | 0.912 | 0.778 | 0.839 | 0.422 | +0.076 |
| Solid | 0.972 | 0.876 | 0.933 | 0.792 | +0.045 |
| Striped | 0.860 | 0.860 | 0.888 | 0.765 | −0.039 |

The Dot gain came from precision and tighter boxes (mAP50-95 0.352 → 0.422); Dot recall did not improve. Besides resolution, batch size, AMP and the PyTorch version also changed. Curves and confusion matrices: `Box*_curve.png`, `confusion_matrix*.png`.

Dot-centre and rail-fitting evaluations were not run for this model.
