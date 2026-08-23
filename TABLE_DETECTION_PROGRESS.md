# Table Detection Progress

Last updated: 2026-08-23  
Current position: Phase 1, Checkpoint 1.6 — define the inference contract and post-processing

This file is the short operational tracker. `TABLE_DETECTION_PLAN.md` contains the explanations, design decisions, and executable checkpoint details.

## Status at a glance

| Area | Status | Evidence |
|---|---|---|
| Dataset preparation and audit | Done | Processed dataset, split manifest, and audit report exist |
| Local YOLO/CUDA environment | Done | Python 3.12, CUDA PyTorch, Ultralytics and OpenCV verified |
| Tiny overfit diagnostic | Done | Representative 10-image subset successfully memorized |
| Two-to-three-epoch smoke tests | Done | YOLO11n tested at 512 and 640; 640 selected |
| Reproducible YOLO11n baseline | Done | Best epoch 46; validation mAP50 0.854 and mAP50-95 0.665 |
| Validation error analysis | Done | Threshold sweep, size metrics, per-image counts, and 20-image gallery generated |
| Detection inference contract | Next | Raw/filtered schema and post-processing rules not implemented |
| Ground-truth-dot rail fitting | Not started | Phase 2 geometry code not implemented |
| Homography and normalized projection | Not started | Depends on verified rail fitting |
| Predicted-dot geometry integration | Not started | Depends on Phase 1 contract and ground-truth geometry |
| Manual four-corner fallback | Not started | Final Phase 2 prototype checkpoint |

## Completed work

- [x] Prepared the five-class YOLO dataset: `Black`, `Cue`, `Dot`, `Solid`, and `Striped`.
- [x] Audited label syntax, class IDs, image/label pairing, counts, and split integrity.
- [x] Removed the confirmed duplicate training label.
- [x] Kept train, validation, and test roles separate.
- [x] Installed the local environment and verified CUDA on the NVIDIA MX550.
- [x] Verified a tiny subset can be overfit before trusting full training.
- [x] Completed 512 and 640 smoke tests; selected 640 for small-object visibility.
- [x] Recorded the baseline configuration in `configs/train.yaml`.
- [x] Trained YOLO11n on 155 training images with early stopping.
- [x] Selected epoch 46 using only the 20-image validation split.
- [x] Independently reloaded and validated `best.pt`.
- [x] Generated standard Ultralytics curves, confusion matrices, and prediction views.
- [x] Added `scripts/analyze_detector_errors.py`.
- [x] Swept confidence thresholds `0.05–0.70` at class-aware IoU `0.50`.
- [x] Generated per-class metrics, box-size recall, per-image dot counts, 20 annotated images, and a worst-cases contact sheet.
- [x] Kept the held-out test split untouched.

## Latest detector evidence

### Standard validation metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.931 | 0.806 | 0.854 | 0.665 |
| Black | 0.959 | 0.850 | 0.878 | 0.765 |
| Cue | 0.974 | 0.850 | 0.977 | 0.838 |
| Dot | 0.833 | 0.640 | 0.640 | 0.308 |
| Solid | 0.943 | 0.859 | 0.888 | 0.695 |
| Striped | 0.947 | 0.832 | 0.884 | 0.720 |

### Error-gallery operating point

Confidence `0.50` maximized Dot F1 among the swept thresholds.

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Black | 17 | 0 | 3 | 1.000 | 0.850 | 0.919 |
| Cue | 17 | 0 | 3 | 1.000 | 0.850 | 0.919 |
| Dot | 236 | 51 | 123 | 0.822 | 0.657 | 0.731 |
| Solid | 83 | 5 | 14 | 0.943 | 0.856 | 0.897 |
| Striped | 73 | 5 | 13 | 0.936 | 0.849 | 0.890 |

What we learned:

- Every labelled dot is in the small-object group below `32 x 32` pixels.
- Small balls are also harder than medium-sized balls.
- Reducing confidence to `0.05` recovers only 15 more matched dots but introduces 181 more false dot predictions than confidence `0.50`.
- Some predicted dots are near the correct location but fail IoU `0.50`; dot-centre error must therefore be measured before deciding whether those predictions are unusable for rail fitting.
- Distant, small-table broadcasts are the clearest failure cases.

## Current artifacts

| Artifact | Location |
|---|---|
| Main plan | `TABLE_DETECTION_PLAN.md` |
| Training configuration | `configs/train.yaml` |
| Best detector | `outputs/detection/baseline-yolo11n-640/weights/best.pt` |
| Baseline plots | `outputs/detection/baseline-yolo11n-640/` |
| Independent validation | `outputs/detection/baseline-yolo11n-640-independent-val/` |
| Error-analysis script | `scripts/analyze_detector_errors.py` |
| Error-gallery report | `outputs/detection/baseline-yolo11n-640-error-analysis/ERROR_GALLERY.md` |
| Worst-cases contact sheet | `outputs/detection/baseline-yolo11n-640-error-analysis/worst_cases.jpg` |
| Threshold sweep | `outputs/detection/baseline-yolo11n-640-error-analysis/threshold_sweep.csv` |
| Per-image dot counts | `outputs/detection/baseline-yolo11n-640-error-analysis/per_image.csv` |

## Next checkpoint — Phase 1.6

- [ ] Define a stable JSON/Python schema for each raw detection.
- [ ] Preserve original-image box and centre coordinates.
- [ ] Separate raw predictions from filtered predictions.
- [ ] Implement conservative ball constraints without hiding detector failures.
- [ ] Derive Dot thresholds from validation evidence rather than choosing them blindly.
- [ ] Add dot-centre-distance diagnostics because Phase 2 consumes dot centres.
- [ ] Run the contract on representative validation images and save readable output.

Checkpoint completion condition: the same image and model checkpoint always produce a documented raw detection list and a traceable filtered list suitable for later geometry input.

## Remaining Phase 2 work

- [ ] Reserve clear full-dot images for geometry development and keep separate sealed geometry cases.
- [ ] Extract ground-truth dot centres in original-image coordinates.
- [ ] Fit and visualize the four rail lines.
- [ ] Define canonical dot ordering on a normalized 2:1 table.
- [ ] Estimate and validate the image-to-table homography.
- [ ] Project ground-truth ball centres into normalized table coordinates.
- [ ] Replace ground-truth inputs with YOLO detections.
- [ ] Add and verify the manual four-corner fallback.
- [ ] Record geometry failure modes before planning robustness work.

## Deferred decisions

- Final per-class confidence thresholds.
- Whether dot-centre matching shows that the current detector is already adequate for geometry.
- Whether the next controlled detector experiment should use higher input resolution or YOLO11s.
- Final prototype acceptance thresholds.
- Performance on the held-out detector test split.
- Generalization to personal-phone images.

