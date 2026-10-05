# Table Detection Progress

Last updated: 2026-10-05
Current position: YOLO11s at 1920 px on a Colab T4 has the best validation metrics so far (Dot recall 0.844, Dot mAP50 0.895, all mAP50-95 0.770); its dot-centre and rail-fitting evaluations are pending, so the 960 px detector remains the repo default

This file is the short operational tracker. `TABLE_DETECTION_PLAN.md` contains the explanations, design decisions, and executable checkpoint details.

## Status at a glance

| Area | Status | Evidence |
|---|---|---|
| Dataset preparation and audit | Done | Processed dataset, split manifest, and audit report exist |
| Local YOLO/CUDA environment | Done | Project `.venv`, Ultralytics 8.4.124, CUDA, and the NVIDIA MX550 successfully completed the resumed 960 px run |
| Tiny overfit diagnostic | Done | Representative 10-image subset successfully memorized |
| Two-to-three-epoch smoke tests | Done | YOLO11n tested at 512 and 640; 640 selected |
| Reproducible YOLO11n baseline | Done | Best epoch 46; validation mAP50 0.854 and mAP50-95 0.665 |
| Higher-resolution YOLO11n retrain | Done | Resumed run stopped normally via patience; best checkpoint at displayed epoch 68, with Dot recall 0.786 and Dot mAP50 0.763 |
| 1280 px YOLO11n retrain (Colab T4) | Trained; dot-centre evaluation pending | Best epoch 55; validation mAP50 0.926, mAP50-95 0.734, Dot mAP50 0.839, Dot recall 0.778; weights in Google Drive |
| 1920 px YOLO11s retrain (Colab T4) | Trained; best so far; dot-centre and rail evaluation pending | Best epoch 27; validation mAP50 0.952, mAP50-95 0.770, Dot mAP50 0.895, Dot recall 0.844; weights in Google Drive |
| Validation error analysis | Done | Threshold sweep, size metrics, per-image counts, and 20-image gallery generated |
| Dot-centre diagnostic | Done on validation | At 960 px and 8 px tolerance, original full frame plus epoch-3 crop dots gives recall 0.955 and F1 0.959; older 640 px metrics are retained below |
| Initial rail-first inference | Evaluated; not promoted | Normalized rail-crop YOLO regressed to 61/359 matched dots versus 324/359 full-frame matches at confidence 0.25 |
| Aspect-preserving crop inference | Evaluated | Without retraining, original full-frame plus crop detections finds 341/359 dots with 14 false positives |
| Crop-aware fine-tuning | Partial training; evaluation complete | 3 of 12 planned epochs completed before memory failures; original full frame plus epoch-3 crop dots finds 343/359 with 13 false positives |
| Detection inference contract | Partial | Rail CLI preserves prediction confidence, centres, inliers, and JSON diagnostics; full ball schema remains |
| Ground-truth-dot rail fitting | Done | Four correct lines recovered on 20/20 validation images; 19/20 pass the clear-image area gate |
| Homography and normalized projection | Not started | Depends on verified rail fitting |
| Predicted-dot geometry integration | Started | Latest fixed-settings 960 px crop comparison: 14/20 original, 16/20 original-plus-crop, and 17/20 original-plus-epoch-3-crop fits are structurally valid; earlier 640 px confidence sweep remains separate |
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
- [x] Resumed the controlled 960 px retrain and allowed Ultralytics early stopping to finish the run normally.
- [x] Selected displayed epoch 68 using only validation fitness (`0.723092`).
- [x] Reloaded and validated the final 960 px `best.pt`; Dot recall improved from `0.640` to `0.786` and Dot mAP50 from `0.640` to `0.763`.
- [x] Evaluated the original 960 px checkpoint and both saved crop-aware checkpoints on all 20 validation images, with centre tolerances 4/8/12 px and a confidence grid.
- [x] Compared normalized rail-first inference with unchanged full-frame YOLO; recorded the regression without promoting it.
- [x] Prepared 775 crop-training examples from 155 originals plus 620 overlapping labelled tiles; retained all five classes and original split membership.
- [x] Completed 3 crop-aware fine-tuning epochs and independently evaluated epoch-1 and epoch-3 checkpoints at 960 px; the planned 12 epochs did not complete.
- [x] Ran unchanged rail fitting on the fixed-settings original/crop/retrained comparisons; recorded validity separately from bed-plane accuracy.
- [x] Verified all 20 unit tests passed and all 175 source image/label pairs plus the original checkpoint remained unchanged after the experiment.
- [ ] Complete a geometry-specific 960 px rail-confidence selection with corner-agreement checks; the crop comparison is not a final geometry operating point.
- [x] Kept the matched-view geometry evaluation set untouched pending an explicit grouped development/evaluation split.
- [x] Kept the held-out test split untouched.

## Latest detector evidence

### Standard validation metrics — YOLO11s at 1920 px on Colab, 2026-10-05

Same recipe as `configs/train_yolo11n_960.yaml` with
`model=yolo11s.pt imgsz=1920 batch=4 nbs=4 workers=2 cache=ram amp=True plots=True`,
on the same Colab T4 setup as the 1280 run. 1920 px is the broadcast frames'
native width, so dots are not downscaled. Early stopping ended the run after 47
epochs in 0.42 hours; the best checkpoint is epoch 27. Metrics are Ultralytics'
final validation of the stripped `best.pt` (19.4 MB, 9.4M parameters). The test
split remains untouched.

| Class | Precision | Recall | mAP50 | mAP50-95 | Δ mAP50 vs 960 |
|---|---:|---:|---:|---:|---:|
| All | 0.938 | 0.929 | 0.952 | 0.770 | +0.060 |
| Black | 0.954 | 0.900 | 0.943 | 0.807 | +0.053 |
| Cue | 0.960 | 1.000 | 0.995 | 0.905 | +0.005 |
| Dot | 0.854 | 0.844 | 0.895 | 0.494 | +0.132 |
| Solid | 0.960 | 0.948 | 0.963 | 0.829 | +0.075 |
| Striped | 0.965 | 0.952 | 0.966 | 0.815 | +0.039 |

This is the first change that raised Dot recall (0.786 → 0.844, about 21 more
of 359 dots); every class improves on the 960 model. Model size and resolution
changed together, so their separate contributions are unknown. Inference costs
37.4 ms per image on the T4, against 9.2 ms for YOLO11n at 1280.

Decision: best candidate, not yet promoted. Next, run
`scripts/analyze_dot_centers.py` at imgsz 1920 with a 16 px tolerance (equal to
8 px at 960) and `scripts/sweep_rail_confidence.py` at imgsz 1920, then copy
the run from Google Drive (`MyDrive/8ballpool/outputs/yolo11s-1920`) into
`outputs/detection/` and make it the script default.

### Standard validation metrics — 1280 px retrain on Colab, 2026-10-05

Same recipe as `configs/train_yolo11n_960.yaml` with
`imgsz=1280 batch=8 nbs=8 workers=2 amp=True plots=True`, trained on a free
Google Colab Tesla T4 (15 GB): Ultralytics 8.4.124, Python 3.13, torch
2.11.0+cu130. Early stopping ended the run after 75 epochs in 0.25 hours; the
best checkpoint is epoch 55. Metrics are Ultralytics' final validation of the
stripped `best.pt`. The test split remains untouched.

| Class | Precision | Recall | mAP50 | mAP50-95 | Δ mAP50 vs 960 |
|---|---:|---:|---:|---:|---:|
| All | 0.913 | 0.873 | 0.926 | 0.734 | +0.034 |
| Black | 0.962 | 0.850 | 0.976 | 0.805 | +0.086 |
| Cue | 0.857 | 1.000 | 0.993 | 0.886 | +0.003 |
| Dot | 0.912 | 0.778 | 0.839 | 0.422 | +0.076 |
| Solid | 0.972 | 0.876 | 0.933 | 0.792 | +0.045 |
| Striped | 0.860 | 0.860 | 0.888 | 0.765 | −0.039 |

The Dot gain comes from precision (+0.044) and tighter boxes (mAP50-95
0.352 → 0.422); Dot recall did not improve (0.786 → 0.778), so the distant and
low-angle misses likely remain. Striped fell on 86 instances, within the noise
of a 20-image split. Resolution was not the only change: batch size, AMP and
the PyTorch version also differ from the 960 run.

Decision: not yet promoted. Next, run `scripts/analyze_dot_centers.py` at
imgsz 1280 with a 10.7 px tolerance (equal to 8 px at 960) against the 960
model's F1 0.933 / recall 0.908 at confidence 0.20, then copy the run from
Google Drive (`MyDrive/8ballpool/outputs/yolo11n-1280`) into
`outputs/detection/`.

### Crop-aware retraining comparison — experiment 2026-09-07

All 20 validation images, 359 labelled dots. Every before/after inference run uses
960 pixels and one-to-one centre matching at 8 px tolerance. Full-frame confidence
0.25 and supplementary crop confidence 0.50 were fixed before training. These are
Dot-centre metrics, not standard box-IoU mAP or sealed-test results.

| Method | TP | FP | FN | Recall | F1 | Structurally valid rail fits |
|---|---:|---:|---:|---:|---:|---:|
| Original 960 px full frame | 324 | 12 | 35 | 90.3% | 0.9324 | 14/20 |
| Original full frame + original crops | 341 | 14 | 18 | 95.0% | 0.9552 | 16/20 |
| Original full frame + epoch-1 crops | 345 | 18 | 14 | 96.1% | 0.9557 | 18/20 |
| Original full frame + epoch-3 crops | 343 | 13 | 16 | 95.5% | 0.9594 | 17/20 |
| Epoch-1 full frame + epoch-1 crops | 347 | 35 | 12 | 96.7% | 0.9366 | 18/20 |
| Epoch-3 full frame + epoch-3 crops | 343 | 31 | 16 | 95.5% | 0.9359 | 17/20 |

Training used 155 original frames plus four overlapping, aspect-preserving tiles
per frame (775 training examples), with all five classes labelled. Only three
epochs completed: epoch 1 at 960 pixels, then epochs 2-3 resumed at 640 with
optimizer state after memory exhaustion. Further memory failures stopped training
during epoch 4. The planned twelve epochs were not completed. `best.pt` is epoch 1;
`last.pt` is epoch 3. Mixed-resolution trainer scores cannot fairly rank them, so
both checkpoints were re-evaluated at the same 960-pixel inference resolution.

Decision: retain the original 960 px full-frame detector. The conservative
experimental supplement is epoch-3 crop detections for **Dot only**: compared with
cropping without retraining, it adds just two matches and removes one false
positive. Most of the gain comes from cropping itself. Replacing the full-frame
detector increases false positives; applying crop supplements to ball classes
also increased several ball-class false-positive counts. No new default model
or production pipeline was promoted.

Epoch-1 supplements offer higher recall and one more valid rail fit, but five
more false positives than epoch 3. The secondary confidence grid also contains a
promising epoch-1 full/crop setting of 0.50/0.50 (345 TP, 9 FP, F1 0.9677); it is
not the preselected primary comparison or a validated production operating point.

Remaining limitations: 11 of the 16 misses with epoch-3 supplements are in one
distant-table image. Four crops add four inference passes, and a mixed-model
variant needs two checkpoints. Rail-fit validity is structural evidence only;
median Dot-oracle corner error is 1.87 px for original crops and 2.10 px for
epoch-3 supplements over different accepted subsets, so improved bed/corner
accuracy has not been established. Validation guided selection; test and geometry
holdout remain untouched. A future training run should keep a consistent high
resolution on hardware with sufficient memory and target small/distant training
examples and labelled hard negatives.

The earlier normalized rail-first path remains unpromoted: crop YOLO found only
61/359 dots with 5 false positives, and the crop hybrid found 92 with 47, at
confidence 0.25. The rectangular crop experiment above bypasses that path's
provisional table locator and rail/lattice filtering.

Evidence: [crop-aware retraining report](outputs/experiments/crop-finetune-v1/REPORT.md),
[matched-image geometry comparison](outputs/experiments/crop-finetune-v1/comparison/REPORT.md),
and [initial rail-first regression report](outputs/geometry/rail-first/validation-960-20260907/REPORT.md).
The remaining sections preserve the earlier baseline results and their original
evaluation settings.

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
| All | 0.915 | 0.875 | 0.892 | 0.704 |
| Black | 0.945 | 0.853 | 0.890 | 0.760 |
| Cue | 0.886 | 1.000 | 0.990 | 0.879 |
| Dot | 0.868 | 0.786 | 0.763 | 0.352 |
| Solid | 0.929 | 0.856 | 0.888 | 0.758 |
| Striped | 0.950 | 0.881 | 0.927 | 0.773 |

Against the 640 px baseline, Dot improved by `+0.035` precision, `+0.146` recall, and `+0.123` mAP50; Dot mAP50-95 improved by `+0.044`. Overall recall improved by `+0.069`, mAP50 by `+0.038`, and mAP50-95 by `+0.039`. These are useful gains on the same validation split, but the per-class changes are noisy because validation contains only 20 images.

The initially interrupted run was resumed from epoch 61. Two attempts encountered Windows system-memory pressure during OpenCV image allocation, but after memory was freed the unchanged run continued and exited successfully through Ultralytics early stopping. The selected checkpoint is displayed epoch 68 (CSV epoch index 67), which had the highest validation fitness (`0.723092`). The metrics above are from Ultralytics' final validation of the stripped `best.pt`; the held-out test split remains untouched.

Decision: promote the 960 px checkpoint as the new Dot-detector candidate, but do not replace the 640-specific geometry confidence `0.05` until Dot-centre and rail-confidence sweeps are repeated with the new model.

### Earlier error-gallery operating point — 640 px baseline

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

### Earlier Dot-centre diagnostic — 640 px baseline

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

## Earlier rail-fitting evidence — 640 px baseline

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
| Historical 640 px detector | `outputs/detection/baseline-yolo11n-640/weights/best.pt` |
| Baseline plots | `outputs/detection/baseline-yolo11n-640/` |
| Independent validation | `outputs/detection/baseline-yolo11n-640-independent-val/` |
| 960 px training configuration | `configs/train_yolo11n_960.yaml` |
| 960 px best detector | `outputs/detection/baseline-yolo11n-960/weights/best.pt` |
| 960 px training log | `outputs/detection/baseline-yolo11n-960/results.csv` |
| 960 px independent validation report | `outputs/detection/baseline-yolo11n-960-independent-val/VALIDATION_REPORT.md` |
| Latest crop-aware results | `outputs/experiments/crop-finetune-v1/REPORT.md` |
| Crop-aware training configuration | `configs/train_yolo11n_crop_finetune.yaml` |
| Crop-aware run status and interruptions | `outputs/experiments/crop-finetune-v1/RUN_NOTES.md` |
| Experimental epoch-3 crop checkpoint | `outputs/detection/yolo11n-960-crop-finetune-v1/weights/last.pt` |
| Experimental epoch-1 checkpoint | `outputs/detection/yolo11n-960-crop-finetune-v1/weights/best.pt` |
| Crop comparison and geometry diagnostics | `outputs/experiments/crop-finetune-v1/comparison/REPORT.md` |
| Initial rail-first regression report | `outputs/geometry/rail-first/validation-960-20260907/REPORT.md` |
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

- [x] Compare original and crop-aware checkpoints using a 960 px Dot-centre distance/confidence grid on all validation images.
- [x] Compare unchanged rail fitting at the fixed full-frame/crop confidences 0.25/0.50.
- [ ] Complete the geometry-specific rail-confidence selection using the unchanged fitter and corner-agreement checks.
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
