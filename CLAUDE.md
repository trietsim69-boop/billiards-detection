# 8-ball pool ball and rail-sight detection

YOLO11 detector for broadcast 8-ball images (classes 0 Black, 1 Cue, 2 Dot = rail sight, 3 Solid, 4 Striped) plus four-rail fitting through the detected dots. This is the final shipped state: there is no Phase 2 (homography, table state) or shot ranking; their plans were deleted and remain in git history.

## Commands

Run from the repo root with `.venv\Scripts\python.exe` (Python 3.12, Ultralytics 8.4.124, torch 2.9 cu126, OpenCV 5). Script defaults are root-relative and point at the best model, `outputs/yolo11s-1920/weights/best.pt` at imgsz 1920.

- Tests: `.venv\Scripts\pytest.exe -q`. `pyproject.toml` puts `scripts/` on `sys.path`, so tests import scripts as bare modules.
- Syntax check: `python -m compileall -q scripts`. There is no build, lint or package install.
- `scripts/analyze_detector_errors.py` holds the shared helpers (`Detection`, label loading, `prf`, `write_csv`, `draw_banner`, `create_contact_sheet`); `scripts/geometry.py` holds `fit_rails`; `sweep_rail_confidence.py` imports from `fit_table_rails.py`. Scripts import Ultralytics inside `main()` so tests and label-mode rail fitting stay torch-free.

## Results

`README.md` reports only the best model and embeds its images straight from `outputs/yolo11s-1920/` (rail-fit overlays, curves, `val_batch0_*.jpg`), so moving or renaming those files breaks the README. Each trained model has `outputs/<model>/RESULTS.md` with weights, settings (`args.yaml`) and evaluations; keep it and the README in sync when a new run or evaluation lands. Compare dot-centre results across image sizes at the same physical tolerance: 8 px at 960 = 10.7 px at 1280 = 16 px at 1920.

## Data rules

- Train only on `data/processed/pix2pockets_v3/data.yaml` (155 train / 20 valid / 20 test / 52 geometry_eval). The raw Roboflow export was removed, and no script regenerates the split, so treat it as immutable.
- Select checkpoints and thresholds on `valid` only; `test` has never been used.

## Hardware

- The local NVIDIA MX550 (2 GB VRAM, low Windows commit memory) can train at most YOLO11n at 960 with batch 1, `workers=0`, `plots=false`, `amp=False`. When free commit memory is low, PyTorch fails to load with a CUDA out-of-memory error or a segfault; closing browser tabs frees it.
- Bigger runs use a free Colab T4 with `configs/train_yolo11s_1920.yaml` (steps in `README.md`). There `amp=True` works, and `nbs` must equal `batch`, or weight decay scales by `batch / nbs`.
- Pass an absolute `project` to `yolo train`; relative ones land under `runs/detect/`.

## Conventions

- Each new run or evaluation goes in a fresh folder under its model's `outputs/<model>/`, which is tracked in git with weights. Commit `best.pt` only, never `last.pt`, and keep files under GitHub's 100 MB limit.
- Keep every image in the denominator; report rejections and false accepts.
- `fit_rails` `valid=True` means structurally plausible, not geometrically exact.
