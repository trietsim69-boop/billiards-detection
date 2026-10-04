# YOLO11n 960 px Independent Validation

Date: 2026-08-24  
Split: `valid` only — 20 images, 582 labelled instances  
Test split: untouched  
Checkpoint: `outputs/detection/baseline-yolo11n-960/weights/best.pt`  
Selected training epoch: 49  
Ultralytics: 8.4.124  
Input size: 960 px  

## Results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| All | 20 | 582 | 0.904 | 0.858 | 0.872 | 0.704 |
| Black | 20 | 20 | 0.943 | 0.824 | 0.842 | 0.754 |
| Cue | 20 | 20 | 0.908 | 0.988 | 0.977 | 0.881 |
| Dot | 20 | 359 | 0.846 | 0.762 | 0.743 | 0.362 |
| Solid | 20 | 97 | 0.958 | 0.845 | 0.886 | 0.747 |
| Striped | 18 | 86 | 0.865 | 0.872 | 0.912 | 0.774 |

## Comparison with the 640 px baseline

| Scope | Metric | 640 px | 960 px | Change |
|---|---|---:|---:|---:|
| Dot | Precision | 0.833 | 0.846 | +0.013 |
| Dot | Recall | 0.640 | 0.762 | +0.122 |
| Dot | mAP50 | 0.640 | 0.743 | +0.103 |
| Dot | mAP50-95 | 0.308 | 0.362 | +0.054 |
| All classes | mAP50 | 0.854 | 0.872 | +0.018 |
| All classes | mAP50-95 | 0.665 | 0.704 | +0.039 |

The 960 px experiment materially improved the small Dot class on the same validation split. The next decision must use the unchanged Dot-centre and rail-fitting diagnostics; a detector metric gain does not by itself prove a better homography input.

## Training note

The model completed 60 epochs. `best.pt` corresponds to epoch 49, which had the highest Ultralytics validation fitness (`0.721643`). Windows commit-memory pressure caused OpenCV allocation failures while beginning later epochs. The selected checkpoint and independent validation were unaffected. Training plots were disabled during resume to reduce host-memory pressure without changing the learned objective.

## Pending

The Dot-centre and rail-confidence sweeps are pending because `.venv/pyvenv.cfg` currently references a missing base Python executable at `C:/Users/DELL/AppData/Local/Programs/Python/Python312/python.exe`.
