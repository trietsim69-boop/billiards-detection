# Table Detection Progress

Last updated: 2026-08-24
Current position: Phase 2, Checkpoint 2.4 — four-rail prototype complete; canonical dot correspondence is next

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
| Dot-centre diagnostic | Done | Best centre F1 0.885; 0.880 recall at confidence 0.25 and 16 px tolerance |
| Detection inference contract | Partial | Rail CLI preserves prediction confidence, centres, inliers, and JSON diagnostics; full ball schema remains |
| Ground-truth-dot rail fitting | Done | Four correct lines recovered on 20/20 validation images; 19/20 pass the clear-image area gate |
| Homography and normalized projection | Not started | Depends on verified rail fitting |
| Predicted-dot geometry integration | Started | Four structurally valid YOLO-driven rail fits on 11/20 validation images at confidence 0.25 |
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
- [x] Added `scripts/analyze_dot_centers.py` and focused one-to-one matching tests.
- [x] Swept Dot-centre tolerances `4`, `8`, `12`, and `16` pixels at the 640-pixel model scale.
- [x] Generated two Dot-centre galleries: the conservative confidence-0.50 diagnostic and the best-F1 confidence-0.25 candidate.
- [x] Added an input-agnostic four-rail fitter in `src/billiards/geometry.py`.
- [x] Added deterministic dominant-line consensus, vertical-safe line equations, opposite-side grouping, convex corners, and explicit failure warnings.
- [x] Added synthetic geometry tests covering perspective, outliers, shuffled points, and insufficient input.
- [x] Added `scripts/fit_table_rails.py` for both labelled images and unlabeled YOLO inference.
- [x] Recovered the correct four rails from ground-truth Dots on all 20 validation images; the distant-table case is rejected only by the configured area gate.
- [x] Ran the unchanged fitter on YOLO Dot centres: 11/20 images produce structurally valid four-rail quadrilaterals.
- [x] Kept the matched-view geometry evaluation set untouched pending an explicit grouped development/evaluation split.
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

### Dot-centre diagnostic

Box IoU understates how useful the current predictions are for geometry. Centre matching uses a one-to-one assignment and measures distance after letterboxing to the model's 640-pixel scale.

| Confidence | Centre tolerance | Precision | Recall | F1 | Images with every labelled dot matched |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 4 px | 0.972 | 0.777 | 0.864 | 7/20 |
| 0.30 | 8 px | 0.908 | 0.850 | 0.878 | 9/20 |
| 0.25 | 16 px | 0.890 | 0.880 | 0.885 | 10/20 |
| 0.05 | 16 px | 0.718 | 0.967 | 0.824 | 14/20 |

What we learned:

- At confidence `0.50`, every match already falls within `4` model-scale pixels; median centre error is `0.44` px and p95 is `1.13` px.
- The confidence-`0.25`, tolerance-`16` row has the best F1 in the tested grid. Of its 316 matches, 301 are within `2` px and 309 are within `4` px, so most accepted centres are very accurate.
- Lowering confidence can recover many geometrically plausible dots, but also introduces false positives. Rail fitting and RANSAC must determine whether those extra candidates help or hurt.
- One distant-table image remains a major failure: only 8 of its 18 labelled dots match at confidence `0.25` and tolerance `16`.
- Decision: retain this YOLO11n checkpoint for the clear/full-dot prototype and begin geometry with ground-truth dots. Do not yet treat it as a robust automatic detector for every broadcast image; revisit higher-resolution training after the geometry stage defines its actual minimum-dot requirements.

## Latest rail-fitting evidence

The same `fit_rails(dot_centres)` function was evaluated with two point sources. No rail was manually drawn.

| Dot source | Four lines recovered | Structurally valid | Interpretation |
|---|---:|---:|---|
| Ground-truth labels | 20/20 | 19/20 | Geometry algorithm works; one distant table is below the 3% image-area gate |
| YOLO at confidence 0.25 | 11/20 | 11/20 | Clear/full-dot prototype works; remaining images usually have only three sufficiently supported rails |

Every successful ground-truth fit has the expected support pattern: six dots on each long rail and three on each short rail, except the single image with one missing label (`6/3/5/3`). Successful YOLO fits may include extra collinear candidates, so their overlays and later homography residuals remain necessary validity checks.

The current safety rule requires at least three supporting points per rail. It intentionally rejects a two-point fourth rail because any two false detections define a line. Missing-dot robustness is deferred until canonical spacing and homography checks can safely constrain that case.

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
| Dot-centre script | `scripts/analyze_dot_centers.py` |
| Dot-centre report | `outputs/detection/baseline-yolo11n-640-dot-centers/DOT_CENTER_REPORT.md` |
| Dot-centre distance sweep | `outputs/detection/baseline-yolo11n-640-dot-centers/distance_sweep.csv` |
| Dot-centre conservative gallery | `outputs/detection/baseline-yolo11n-640-dot-centers/gallery/` |
| Dot-centre best-F1 candidate gallery | `outputs/detection/baseline-yolo11n-640-dot-centers-conf025/gallery/` |
| Geometry configuration | `configs/geometry.yaml` |
| Rail-fitting module | `src/billiards/geometry.py` |
| Rail-fitting CLI | `scripts/fit_table_rails.py` |
| Geometry tests | `tests/test_geometry.py` |
| Ground-truth rail report | `outputs/geometry/validation-ground-truth-rails/RAIL_FITTING_REPORT.md` |
| Ground-truth rail contact sheet | `outputs/geometry/validation-ground-truth-rails/contact_sheet.jpg` |
| YOLO rail report | `outputs/geometry/validation-yolo-rails/RAIL_FITTING_REPORT.md` |
| YOLO rail contact sheet | `outputs/geometry/validation-yolo-rails/contact_sheet.jpg` |
| Real-image testing guide | `REAL_IMAGE_TESTING.md` |

## Next checkpoint — Phase 1.6

- [ ] Define a stable JSON/Python schema for each raw detection.
- [ ] Preserve original-image box and centre coordinates.
- [ ] Separate raw predictions from filtered predictions.
- [ ] Implement conservative ball constraints without hiding detector failures.
- [ ] Derive Dot thresholds from validation evidence rather than choosing them blindly.
- [x] Add dot-centre-distance diagnostics because Phase 2 consumes dot centres.
- [ ] Run the contract on representative validation images and save readable output.

Checkpoint completion condition: the same image and model checkpoint always produce a documented raw detection list and a traceable filtered list suitable for later geometry input.

## Remaining Phase 2 work

- [ ] Reserve clear full-dot images for geometry development and keep separate sealed geometry cases.
- [x] Extract ground-truth dot centres in original-image coordinates.
- [x] Fit and visualize the four rail lines.
- [ ] Define canonical dot ordering on a normalized 2:1 table.
- [ ] Estimate and validate the image-to-table homography.
- [ ] Project ground-truth ball centres into normalized table coordinates.
- [ ] Replace ground-truth inputs with YOLO detections.
- [ ] Add and verify the manual four-corner fallback.
- [ ] Record geometry failure modes before planning robustness work.

## Deferred decisions

- Final per-class confidence thresholds.
- Exact minimum per-rail dot coverage required for stable line fitting; the current detector is adequate only for the initial clear/full-dot prototype.
- Whether the next controlled detector experiment should use higher input resolution or YOLO11s.
- Final prototype acceptance thresholds.
- Performance on the held-out detector test split.
- Generalization to personal-phone images.
