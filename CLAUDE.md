# Billiards table-state detection

Single broadcast image of an 8-ball table → YOLO detections (balls + rail sights) → four rail lines → bed homography → normalized 2:1 table state. Shot ranking comes later. Research prototype on Windows with a small local GPU.

Status (2026-10-04): Phase 1 (detector) is nearly closed; the detection JSON contract and the 960 Dot-confidence selection remain open. Phase 2 (geometry) has rail fitting only; homography, table state and the manual four-corner fallback are not started. Next work is `tasks/todo.md` Tasks 1–3.

## Commands

Run from the repo root with `.venv\Scripts\python.exe` (Python 3.12, Ultralytics 8.4.124, torch 2.9 cu126, OpenCV 5). Script defaults are root-relative paths and point at the 960 model.

- Tests: `.venv\Scripts\pytest.exe -q`. `pyproject.toml` puts `src/` and `scripts/` on `sys.path`, so tests import scripts as bare modules. 9/9 pass as of 2026-10-04.
- Syntax check: `python -m compileall -q src scripts`. There is no build, lint or package install.
- Train: `.venv\Scripts\yolo.exe detect train cfg=configs/train_yolo11n_960.yaml project="$PWD\outputs\detection" name=<new-name>`. Use an absolute `project`; relative ones landed under `runs\detect\`.
- `scripts/analyze_detector_errors.py` holds the shared helpers (`Detection`, label loading, `prf`, `write_csv`, `draw_banner`, `create_contact_sheet`); `sweep_rail_confidence.py` imports from `fit_table_rails.py`. Scripts import Ultralytics inside `main()` so tests and label-mode rail fitting stay torch-free.

## Best results so far

Full per-method tables are in `README.md`; validation split, 20 images.

- **Best trained detector:** `outputs/detection/baseline-yolo11n-960/weights/best.pt` from `configs/train_yolo11n_960.yaml`. Recipe: pretrained `yolo11n.pt`, imgsz 960, batch 1 / nbs 2, patience 20, `amp=False`, seed 42 deterministic, mosaic/scale/translate/mixup off, only `fliplr 0.5` + mild HSV. Epoch 68: mAP50 0.892, mAP50-95 0.704, Dot recall 0.786, Dot mAP50 0.763. The 640 baseline reached 0.854 / 0.665 / Dot recall 0.640.
- **Dot centres (8 px at 960, 359 dots):** full frame at conf 0.25 → 324 matched, F1 0.932. Adding four 65% tiles at conf 0.50 (Dot only) → 341, F1 0.955. Tiles from the crop fine-tune → 343, F1 0.959. Tiling stayed experimental.
- **Rail fitting:** labelled dots → 19/20 valid. 640 model at conf 0.05 → 19/20 valid and agreeing with labels. `configs/geometry.yaml` now runs at imgsz 960 with that 640-derived 0.05; re-run `sweep_rail_confidence.py` to select the 960 threshold.
- **Dead ends:** crop-aware fine-tuning (3 of 12 epochs, forced 960→640 by memory, +2 dots over plain tiling) and rectified rail-strip inference (61/359 dots). Untried: YOLO11s, AMP, hard-negative mining.

The code for tiling, crop fine-tuning, rail-first detection, table localization and the tiny-overfit check was removed; recover it from commit `d6eb42e` (`git show d6eb42e:<path>`). `tasks/` and the plan docs still reference those files.

## Data rules

- Train only on `data/processed/pix2pockets_v3/data.yaml`; the raw Roboflow `data.yaml` points at split dirs that do not exist. Both trees are immutable inputs; generated datasets go under `outputs/datasets/`.
- Classes in order: 0 Black, 1 Cue, 2 Dot, 3 Solid, 4 Striped. "Dot" is a rail sight (diamond).
- Splits: train 155 / valid 20 / test 20 / geometry_eval 52 (`a`/`f`/`t` matched views of 25 situations). Select checkpoints, thresholds and settings on `valid` only. `test` stays sealed until detector choices are frozen; geometry_eval stays untouched until the grouped dev/holdout manifest is frozen (Task 5).
- `data/processed/pix2pockets_v3/README.md` names `scripts/prepare_dataset.py`, which is missing: the processed split cannot currently be regenerated.

## Hardware limits

NVIDIA MX550 with 2 GB VRAM, Windows with low commit headroom. 960 px training needs batch 1, `workers=0`, `cache=false`, `plots=false`; long runs hit OpenCV/cuDNN host-allocation failures and resume from `last.pt`. Keep `amp=False`: Ultralytics' AMP check fails on this GPU.

## Geometry conventions

- Two-plane rule: rail sights sit on the raised rail cap, above the playable bed. A sight-derived transform is never the bed homography; ball projection uses reviewed inner-cushion corners. Ball box centres are not bed contact points.
- `fit_rails` (`src/billiards/geometry.py`) is detector-agnostic and takes `PointObservation`s from labels or YOLO. `valid=True` means structurally plausible, not geometrically correct.
- Score dots by one-to-one centre distance at model scale (`scripts/analyze_dot_centers.py`); box IoU understates tiny Dot boxes.

## Experiment conventions

- Each run writes to a fresh directory under `outputs/`, which is tracked in git, weights included. GitHub rejects files over 100 MB, so keep generated datasets and large galleries out of commits. Save weights SHA256, config and a source snapshot with the results.
- Keep every image in the denominator; report rejections and false accepts. Freeze thresholds before any holdout run.
- `TABLE_DETECTION_PROGRESS.md` is authoritative for results. Only the 960 artifacts were kept in `outputs/`, so its links to 640, crop and rail-first evidence are dead; the recorded numbers stand. `baseline-yolo11n-960-independent-val/VALIDATION_REPORT.md` predates the resume and cites epoch 49.

## Docs

- `README.md`: results by method, setup and usage.
- `tasks/plan.md`, `tasks/todo.md`: read before starting work — ordered tasks with acceptance criteria.
- `TABLE_DETECTION_PROGRESS.md`: read when you need a metric, checkpoint path or past decision.
- `TABLE_DETECTION_PLAN.md`: Phase 1/2 checkpoint definitions and prototype gates.
- `RAIL_FIRST_DETECTION_PLAN.md`: the abandoned rail-first branch and the source of the two-plane rule.
- `REAL_IMAGE_TESTING.md`: running rail fitting on new photos.
- `TECHNICAL_PLAN.md`, `ARCHITECTURE.md`: original MVP vision; the module layout they describe (`cli.py`, `detection.py`, `shots.py`, …) does not exist yet.
