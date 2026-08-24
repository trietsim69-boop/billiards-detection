# Table Detection Progress

Last updated: 2026-08-24
Current position: 960 px Dot-focused retrain complete; repeat downstream Dot-centre and rail sweeps before canonical correspondence

This file is the short operational tracker. `TABLE_DETECTION_PLAN.md` contains the explanations, design decisions, and executable checkpoint details.

## Status at a glance

| Area | Status | Evidence |
|---|---|---|
| Dataset preparation and audit | Done | Processed dataset, split manifest, and audit report exist |
| Local YOLO/CUDA environment | Needs repair | Training and validation completed, but `.venv` now references a missing base Python executable; custom diagnostics are paused |
| Tiny overfit diagnostic | Done | Representative 10-image subset successfully memorized |
| Two-to-three-epoch smoke tests | Done | YOLO11n tested at 512 and 640; 640 selected |
| Reproducible YOLO11n baseline | Done | Best epoch 46; validation mAP50 0.854 and mAP50-95 0.665 |
| Higher-resolution YOLO11n retrain | Done (resource-limited) | 60 epochs completed; 960 px candidate selected at epoch 49; Dot recall 0.762 and Dot mAP50 0.743 |
| Validation error analysis | Done | Threshold sweep, size metrics, per-image counts, and 20-image gallery generated |
| Dot-centre diagnostic | Done | Best centre F1 0.885; 0.880 recall at confidence 0.25 and 16 px tolerance |
| Detection inference contract | Partial | Rail CLI preserves prediction confidence, centres, inliers, and JSON diagnostics; full ball schema remains |
| Ground-truth-dot rail fitting | Done | Four correct lines recovered on 20/20 validation images; 19/20 pass the clear-image area gate |
| Homography and normalized projection | Not started | Depends on verified rail fitting |
| Predicted-dot geometry integration | Started | At Dot confidence 0.05, four rails are produced on 20/20 validation images and 19/20 pass structure plus label agreement |
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
- [x] Ran the unchanged fitter on YOLO Dot centres at confidence `0.25`: 11/20 images produce structurally valid four-rail quadrilaterals.
- [x] Added a downstream confidence sweep from `0.05` to `0.50`, using one YOLO pass and ground-truth corner agreement.
- [x] Selected confidence `0.05` as the provisional Dot-candidate threshold: 20/20 produce four rails and 19/20 are structurally valid and agree with labelled corners.
- [x] Smoke-tested YOLO11n at 960 px for 3 epochs with batch 1; peak reported GPU memory was about 0.68 GB.
- [x] Ran the controlled 960 px retrain for 60 complete epochs and selected epoch 49 using only validation fitness.
- [x] Independently reloaded and validated the 960 px `best.pt`; Dot recall improved from `0.640` to `0.762`.
- [ ] Repeat the Dot-centre and downstream rail-confidence sweeps with the 960 px checkpoint after repairing the local Python launcher.
- [x] Kept the matched-view geometry evaluation set untouched pending an explicit grouped development/evaluation split.
- [x] Kept the held-out test split untouched.

## Latest detector evidence

### Standard validation metrics — 640 px baseline

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.931 | 0.806 | 0.854 | 0.665 |
| Black | 0.959 | 0.850 | 0.878 | 0.765 |
| Cue | 0.974 | 0.850 | 0.977 | 0.838 |
| Dot | 0.833 | 0.640 | 0.640 | 0.308 |
| Solid | 0.943 | 0.859 | 0.888 | 0.695 |
| Striped | 0.947 | 0.832 | 0.884 | 0.720 |

### Standard validation metrics — 960 px retrain

The controlled change was input resolution `640 -> 960`; YOLO11n, the train/validation split, pretrained initialization, augmentation policy, seed, and validation data remained the same. Batch `1` was used as the memory-safe local setting. The test split remains untouched.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.904 | 0.858 | 0.872 | 0.704 |
| Black | 0.943 | 0.824 | 0.842 | 0.754 |
| Cue | 0.908 | 0.988 | 0.977 | 0.881 |
| Dot | 0.846 | 0.762 | 0.743 | 0.362 |
| Solid | 0.958 | 0.845 | 0.886 | 0.747 |
| Striped | 0.865 | 0.872 | 0.912 | 0.774 |

Dot improved by `+0.013` precision, `+0.122` recall, `+0.103` mAP50, and `+0.054` mAP50-95. Overall mAP50-95 improved by `+0.039`. These are useful gains on the same validation split, but the per-class changes are noisy because validation contains only 20 images.

Training completed through epoch 60. Repeated Windows commit-memory pressure caused OpenCV image-buffer allocation errors while starting later epochs, so the run was stopped rather than changing AMP or resolution mid-experiment. The chosen epoch-49 checkpoint predates those failures and had the highest Ultralytics validation fitness (`0.721643`). Independent validation reproduced the class metrics above.

Decision: promote the 960 px checkpoint as the new Dot-detector candidate, but do not replace the 640-specific geometry confidence `0.05` until Dot-centre and rail-confidence sweeps are repeated with the new model.

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
| YOLO at confidence 0.25 | 11/20 | 11/20 | Detector-F1 operating point removes too many rail candidates |
| YOLO at confidence 0.05 | 20/20 | 19/20 | RANSAC recovers the correct rails; the same distant table remains below the area gate |

Every successful ground-truth fit has the expected support pattern: six dots on each long rail and three on each short rail, except the single image with one missing label (`6/3/5/3`). Successful YOLO fits may include extra collinear candidates, so their overlays and later homography residuals remain necessary validity checks.

At confidence `0.05`, 19/20 fitted quadrilaterals pass both structural validation and ground-truth corner agreement. Their median corner error is `2.89` original-image pixels. The remaining image is not a wrong accepted result: it is explicitly rejected because the distant table occupies only `1.6%` of the image.

The current safety rule requires at least three supporting points per rail. Low-confidence Dot candidates are retained for geometry, then line consensus rejects spatial outliers. This is why the best detector-F1 threshold (`0.25`) is not the best downstream rail threshold (`0.05`).

The 960 px retrain has now improved standard Dot metrics. The rail results above still belong to the 640 px model, so confidence `0.05` remains provisional until the same geometry sweep is repeated with the new checkpoint.

## Current artifacts

| Artifact | Location |
|---|---|
| Main plan | `TABLE_DETECTION_PLAN.md` |
| Training configuration | `configs/train.yaml` |
| Best detector | `outputs/detection/baseline-yolo11n-640/weights/best.pt` |
| Baseline plots | `outputs/detection/baseline-yolo11n-640/` |
| Independent validation | `outputs/detection/baseline-yolo11n-640-independent-val/` |
| 960 px training configuration | `configs/train_yolo11n_960.yaml` |
| 960 px best detector | `outputs/detection/baseline-yolo11n-960/weights/best.pt` |
| 960 px training log | `outputs/detection/baseline-yolo11n-960/results.csv` |
| 960 px independent validation report | `outputs/detection/baseline-yolo11n-960-independent-val/VALIDATION_REPORT.md` |
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
| Rail-confidence sweep script | `scripts/sweep_rail_confidence.py` |
| Rail-confidence sweep report | `outputs/geometry/rail-confidence-sweep/RAIL_CONFIDENCE_SWEEP.md` |
| Rail-confidence sweep graph | `outputs/geometry/rail-confidence-sweep/confidence_sweep.png` |
| Real-image testing guide | `REAL_IMAGE_TESTING.md` |

## Next checkpoint — compare 960 px downstream utility

- [ ] Repair or recreate `.venv` so its Python launcher resolves correctly.
- [ ] Repeat the Dot-centre distance/confidence sweep at the 960 px model scale.
- [ ] Repeat the rail-confidence sweep using the unchanged geometry fitter.
- [ ] Compare the 640 and 960 models on rail success, corner error, and failure cases.
- [ ] Select the detector checkpoint and Dot confidence used by canonical correspondence.

Checkpoint completion condition: one detector/confidence pair is selected from both detector metrics and unchanged downstream geometry evidence.

## Detection contract checkpoint — Phase 1.6

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

- Final per-class confidence thresholds; Dot uses provisional geometry-specific confidence `0.05`.
- Exact minimum per-rail dot coverage required for stable line fitting; the current detector is adequate only for the initial clear/full-dot prototype.
- Whether YOLO11s is worth testing after the 960 px YOLO11n downstream comparison.
- Final prototype acceptance thresholds.
- Performance on the held-out detector test split.
- Generalization to personal-phone images.
