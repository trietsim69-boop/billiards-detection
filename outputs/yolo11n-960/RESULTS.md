# YOLO11n at 960 px — laptop baseline

The first model trained to completion on the local NVIDIA MX550 (2 GB VRAM). It is superseded by [YOLO11s at 1920](../yolo11s-1920/RESULTS.md).

## Training

Pretrained `yolo11n.pt`, imgsz 960, batch 1 / nbs 2, `amp=False`, `workers=0`, `plots=false`, seed 42, deterministic. Mosaic, scale, translate and mixup off; horizontal flips and mild HSV jitter only. 87 epochs including a resume after Windows memory failures; early stopping kept epoch 68. Exact settings: [`args.yaml`](args.yaml); per-epoch metrics: [`results.csv`](results.csv).

## Validation (20 images, Ultralytics `val` of `best.pt`)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.915 | 0.875 | 0.892 | 0.704 |
| Black | 0.945 | 0.853 | 0.890 | 0.760 |
| Cue | 0.886 | 1.000 | 0.990 | 0.879 |
| Dot | 0.868 | 0.786 | 0.763 | 0.352 |
| Solid | 0.929 | 0.856 | 0.888 | 0.758 |
| Striped | 0.950 | 0.881 | 0.927 | 0.773 |

## Dot centres

359 labelled dots, one-to-one matching within 8 px at 960 scale (16 px in the original 1920 px frame). Details: [`dot-centers/`](dot-centers/DOT_CENTER_REPORT.md).

| Confidence | Found | False positives | Missed | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 332 | 40 | 27 | 0.892 | 0.925 | 0.908 |
| 0.20 | 326 | 14 | 33 | 0.959 | 0.908 | **0.933** |
| 0.25 | 324 | 12 | 35 | 0.964 | 0.903 | 0.932 |
| 0.50 | 314 | 4 | 45 | 0.987 | 0.875 | 0.928 |

17 of the 45 misses at confidence 0.50 are in one distant-table image (`2_png`).

## Other evidence

- [`error-analysis/`](error-analysis/ERROR_GALLERY.md): per-class box-IoU errors at the best Dot threshold (0.40: Dot F1 0.824), a 20-image gallery and a worst-cases sheet.
- [`photo-rail-fit/`](photo-rail-fit/RAIL_FITTING_REPORT.md): one personal phone photo. The model found only 11 Dot candidates, below the 12 that rail fitting needs, so no rails were fitted. The model was trained on broadcast frames only.
- Rail fitting was not swept for this model.
