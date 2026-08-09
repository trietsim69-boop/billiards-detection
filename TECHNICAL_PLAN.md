# 8-Ball Pool State Detection and Shot Ranking — Technical Plan

Status: approved MVP plan; implementation has not started  
Last updated: 2026-08-08

## 1. Objective

Build a local Python application that accepts a single photograph of an 8-ball pool table, detects the balls and rail diamonds, converts the detections into a normalized top-down table state, and returns the best plausible direct shots as both an annotated image and structured JSON.

The project should demonstrate an end-to-end computer-vision and geometry pipeline without requiring a production-grade physics engine or reinforcement learning.

### MVP success criteria

- Detect `Black`, `Cue`, `Dot`, `Solid`, and `Striped` objects in varied single-table images.
- Estimate a valid table homography automatically when enough rail dots are visible.
- Provide a manual four-corner calibration fallback when automatic geometry fails.
- Transform ball positions into a canonical 2:1 table coordinate system.
- Generate legal direct-shot candidates for a selected player group.
- Reject shots with blocked cue-ball or object-ball paths.
- Rank candidates using angle, distance, clearance, and approximate scratch risk.
- Save an annotated source image, a top-down diagnostic image, and a JSON result.
- Run through a local command-line interface.

## 2. Confirmed MVP scope

### Included

- American 8-ball.
- One still image per inference request.
- Arbitrary viewpoints when the table diamonds are sufficiently visible.
- Automatic table reconstruction from the diamond pattern.
- Manual calibration fallback.
- Direct object-ball-to-pocket shots.
- Deterministic geometric ranking.
- Local CLI output.

### Explicitly deferred

- Live video and temporal tracking.
- Reinforcement learning.
- Shot-power prediction.
- Spin, masse, jump shots, throw, and detailed cushion physics.
- Bank and kick shots.
- Full strategic planning across multiple future turns.
- Production web or mobile interface.
- Guaranteed support for tables without visible diamonds.
- Exact real-world measurements in centimetres.

## 3. Current repository and data audit

Repository root:

```text
D:\billiards project
```

Current dataset:

```text
8-Ball Pool.v3i.yolov11/
```

The dataset is the public Pix2Pockets/BachelorThesis Roboflow export, version 3, licensed CC BY 4.0.

### Confirmed dataset contents

- 247 images and 247 label files.
- Five classes: `Black`, `Cue`, `Dot`, `Solid`, and `Striped`.
- 195 normally numbered images used as the main detection set.
- 52 images with `a`, `f`, or `t` suffixes representing 25 matched table situations from angled, front, and top views.
- No Roboflow augmentation was applied to this export.
- All images are currently under `train/`; the referenced `valid/` and `test/` directories do not exist.
- One malformed mixed-format annotation exists in the label for `18t`: a `Dot` segmentation polygon appears in an otherwise bounding-box dataset. It must be converted to a bounding box or corrected from the source annotation before training.

Observed annotation counts:

| Class | Instances |
|---|---:|
| Black | 242 |
| Cue | 245 |
| Dot | 4,426 |
| Solid | 1,225 |
| Striped | 1,105 |

### Data implications

The downloaded `data.yaml` cannot be used as-is for meaningful training because it has no validation or test data. The 52 matched multi-view images are particularly valuable for evaluating homography accuracy and must not be randomly scattered across training and evaluation.

The raw export will be treated as immutable. Any repaired labels and generated splits will be written to a separate processed-data directory with a manifest describing their source.

## 4. System architecture

```text
Input image
    -> input validation and letterbox resize
    -> YOLO11 ball-and-dot detector
    -> detection post-processing
    -> table-line and homography estimation
    -> normalized top-down table state
    -> legal direct-shot generation
    -> deterministic shot ranking
    -> annotated image + top-down image + JSON
```

### Coordinate systems

The application will keep coordinate systems explicit:

1. **Image coordinates:** pixel positions in the original input image.
2. **Model coordinates:** positions in the letterboxed detector input.
3. **Table coordinates:** normalized coordinates with table width `x in [0, 2]` and height `y in [0, 1]`.
4. **Render coordinates:** an optional top-down canvas, initially `1000 x 500` pixels.

The six canonical pocket centres are:

```text
(0, 0)   (1, 0)   (2, 0)
(0, 1)   (1, 1)   (2, 1)
```

Exact feet or centimetres are unnecessary for MVP geometry. Pocket radius and ball radius will be represented relative to the normalized table and kept configurable.

## 5. Proposed repository structure

```text
D:\billiards project\
├── TECHNICAL_PLAN.md
├── README.md
├── pyproject.toml
├── .gitignore
├── configs\
│   ├── train.yaml
│   └── ranking.yaml
├── src\
│   └── billiards\
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── detection.py
│       ├── geometry.py
│       ├── calibration.py
│       ├── state.py
│       ├── shots.py
│       ├── ranking.py
│       ├── rendering.py
│       └── schemas.py
├── scripts\
│   ├── audit_dataset.py
│   ├── prepare_dataset.py
│   └── train_detector.py
├── tests\
│   ├── test_dataset.py
│   ├── test_geometry.py
│   ├── test_shots.py
│   └── fixtures\
├── data\
│   └── processed\
├── models\
├── outputs\
└── 8-Ball Pool.v3i.yolov11\
```

The existing dataset directory will remain unchanged initially. Generated datasets, model weights, training runs, and inference output should be excluded from ordinary Git tracking unless an explicit storage decision is made later.

## 6. Implementation plan

### Milestone 0 — Project bootstrap and reproducibility

Deliverables:

- Python project configuration and locked dependency versions.
- CLI entry point.
- Configuration loading.
- Structured logging.
- Seed control for repeatable splits and training.
- Unit-test setup.
- `.gitignore` entries for generated data, weights, runs, caches, and output images.

Initial dependencies:

- Python 3.11 or a compatible supported version.
- Ultralytics YOLO11.
- OpenCV.
- NumPy.
- PyYAML.
- Pydantic or dataclasses for result schemas.
- Pytest.
- Pillow for diagnostics where useful.

Training should support either a local CUDA-capable GPU or a hosted notebook. Inference must remain usable on CPU with a small YOLO11 model.

### Milestone 1 — Dataset repair and split

Create `scripts/audit_dataset.py` to verify:

- Every image has one corresponding label file.
- Class identifiers are in the configured range.
- Each detection row has exactly five YOLO object-detection fields.
- Coordinates and dimensions are normalized and valid.
- Empty labels are reported.
- Exact and perceptual duplicates are reported.
- Images can be decoded.
- The 195 main images and 52 multi-view images are identified correctly.

Create `scripts/prepare_dataset.py` to:

1. Copy or link the raw inputs into a generated processed dataset without modifying the source export.
2. Convert the polygon annotation in `18t` into its tight bounding box after visual verification.
3. Reserve the 52 suffixed multi-view images as a geometry evaluation set grouped by their 25 situation identifiers.
4. Split the 195 main images into approximately 155 training, 20 validation, and 20 detection-test images, matching the paper's evaluation size.
5. Inspect for visually related frames using perceptual hashes and keep each near-duplicate group in one split.
6. Write `split_manifest.csv` containing source filename, scene group, split, and any repair performed.
7. Generate a valid `data.yaml` with paths relative to the processed dataset.
8. Produce class-frequency and sample-contact-sheet reports.

Acceptance checks:

- Re-running with the same seed produces exactly the same manifest.
- No scene or near-duplicate group crosses a split boundary.
- Ultralytics can load every split.
- The raw dataset remains byte-for-byte unchanged.

### Milestone 2 — Ball and dot detector

Start from a pretrained YOLO11 nano model to keep iteration fast. Compare a small model only if the nano model misses too many small balls or dots.

Initial training configuration:

- Input size: `640`.
- Pretrained weights: YOLO11n.
- Epochs: up to `100`, with early stopping.
- Batch size: automatically selected for available memory.
- Save the best validation checkpoint by mAP50-95.
- Preserve aspect ratio with letterboxing; do not stretch images.
- Use mild brightness, contrast, scale, and perspective augmentation.
- Avoid aggressive crops because the rail-dot layout is necessary for geometry.

Track at minimum:

- Per-class precision and recall.
- Per-class AP50 and AP50-95.
- Confusion matrix.
- False negatives when balls are small.
- Dot recall per rail, not only aggregate dot AP.

Post-processing rules:

- Class-agnostic NMS for highly overlapping ball predictions.
- At most one cue ball and one black ball, keeping the highest-confidence candidate.
- Reject implausible dot sizes relative to the median dot detection.
- Allow geometric RANSAC to reject spatial dot outliers rather than relying only on confidence thresholds.

The first detector milestone is complete when it produces stable detections on held-out images and enough dots to recover the table in most unobstructed evaluation images.

### Milestone 3 — Automatic table geometry

The automatic method will follow the paper's core idea while keeping the implementation independently testable.

Processing steps:

1. Convert each `Dot` bounding box into a centre point.
2. Fit dominant rail lines using a vertical-line-safe RANSAC representation of `a*x + b*y + c = 0`.
3. Remove inliers and repeat until four plausible rail lines are obtained.
4. Group the lines into opposite sides and intersect adjacent sides to form a convex quadrilateral.
5. Reject invalid configurations using convexity, minimum area, point support, and reprojection-error checks.
6. Determine long-side and short-side orientation from the detected diamond counts and spacing.
7. Match rail dots to their canonical positions.
8. Add the four virtual corner intersections to the correspondence set.
9. Estimate the image-to-table homography with `cv2.findHomography(..., RANSAC)`.
10. Report a geometry confidence score and diagnostics.

Expected canonical dot positions:

- Long rails: six dots at `x = 0.25, 0.50, 0.75, 1.25, 1.50, 1.75` on `y = 0` and `y = 1`.
- Short rails: three dots at `y = 0.25, 0.50, 0.75` on `x = 0` and `x = 2`.

These positions must be visually confirmed against the dataset before being fixed in code.

Geometry quality fields:

- Number of accepted dots.
- Number of supported rail lines.
- Inlier ratio.
- Mean and maximum reprojection error.
- Convexity and table-area checks.
- Automatic or manual calibration source.

#### Manual fallback

Provide a calibration command that displays the source image and asks the user to click four ordered inner table corners. The resulting homography is saved in a JSON calibration file that can be reused for images from the same camera.

Manual calibration is also the preferred mode for a permanently mounted camera.

### Milestone 4 — Ball positions and table-state representation

For each accepted ball detection:

1. Establish its image anchor point.
2. Transform that point through the homography.
3. Reject points outside the table plus a small tolerance.
4. Estimate a robust normalized ball radius from the transformed detections or use a configured fallback.
5. Preserve detection confidence and original bounding-box information.

The baseline anchor will be the bounding-box centre. A later refinement will implement the paper's angled-view heuristic, moving the anchor from the centre toward the visually appropriate top portion of the box as a function of the estimated view angle. The baseline and refined methods will be compared on the 25 multi-view situations.

Proposed JSON structure:

```json
{
  "image": "example.jpg",
  "geometry": {
    "method": "dots",
    "confidence": 0.91,
    "homography": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
    "reprojection_error": 0.01
  },
  "player_group": "solid",
  "balls": [
    {
      "id": "ball-001",
      "class": "Cue",
      "confidence": 0.97,
      "table_xy": [0.42, 0.61]
    }
  ],
  "shots": []
}
```

The actual homography values above are placeholders illustrating the schema.

### Milestone 5 — Direct-shot generation

The player group will be supplied through the CLI as `solid`, `striped`, or `open`. The engine will not try to infer whose turn it is from an image.

Legality rules for MVP:

- With an assigned group, only that group is targetable while any of its balls remain.
- The black ball becomes targetable only after the assigned group has been cleared.
- With an open table, generate candidates for both solids and stripes but keep their rankings separated.
- Missing or ambiguous cue/black detections produce an explicit warning rather than a silent assumption.

For every legal object-ball and pocket pair:

1. Compute the object-ball-to-pocket direction.
2. Compute the ghost-ball centre one ball diameter behind the object ball.
3. Confirm that the ghost-ball centre lies within playable bounds.
4. Test the cue-ball-to-ghost-ball segment for collision with every non-participating ball.
5. Test the object-ball-to-pocket segment for collision with every non-participating ball.
6. Measure minimum path clearance, cut angle, cue travel distance, and object travel distance.
7. Reject candidates that violate configurable safety margins.

Collision checks will use point-to-segment distance with a threshold based on two ball radii plus a small uncertainty margin.

### Milestone 6 — Shot ranking

Use an interpretable weighted score rather than RL. Initial ranking features:

- Cut-angle difficulty.
- Object-ball distance to pocket.
- Cue-ball distance to ghost-ball position.
- Minimum clearance from obstacles.
- Pocket type, allowing side-pocket and corner-pocket difficulty to differ.
- Approximate scratch risk.
- Detection and geometry confidence.

Weights will live in `configs/ranking.yaml`, not in source code. Initial weights are provisional and will be tuned using hand-labelled preference examples.

The scratch-risk heuristic will initially penalize shots where the cue ball's approximate post-impact continuation or tangent path leads close to a pocket. It will be clearly labelled approximate because spin and detailed collision physics are outside MVP scope.

Output the best three candidates with a score breakdown so the ranking remains explainable.

### Milestone 7 — Rendering and CLI

Proposed commands:

```powershell
python -m billiards.cli audit-data
python -m billiards.cli prepare-data --seed 42
python -m billiards.cli train --config configs/train.yaml
python -m billiards.cli calibrate --image path\to\image.jpg
python -m billiards.cli detect --image path\to\image.jpg
python -m billiards.cli analyze --image path\to\image.jpg --group solid --top-k 3
```

`analyze` should write:

- `result.json` with detections, geometry, candidates, scores, and warnings.
- `annotated.jpg` with boxes, rail lines, table outline, and suggested shot paths.
- `top_down.jpg` with the normalized state and ranked candidates.

Exit codes should distinguish success, low-confidence geometry, missing required detections, and invalid input.

## 7. Validation strategy

### Detector evaluation

- Evaluate once on the untouched 20-image detection test set after model selection.
- Report all five classes separately.
- Include qualitative failure galleries.
- Track performance against ball pixel size and table coverage.

### Homography evaluation

Use the 52 held-out multi-view images representing 25 situations:

1. Treat each top-view image as the reference state.
2. Estimate homographies independently for its paired angled/front views.
3. Match balls by class and spatial assignment.
4. Measure normalized table-coordinate displacement.
5. Report median, mean, 90th percentile, and worst-case error.

Metric centimetres may be reported only if a documented physical table scale is introduced. Normalized coordinate error is the authoritative project metric.

### Shot-engine tests

Create synthetic table-state fixtures covering:

- Straight unobstructed pots.
- Blocked cue paths.
- Blocked object paths.
- Thin cuts.
- Balls close to cushions.
- Illegal early black-ball attempts.
- Open-table states.
- No available direct shot.
- Scratch-prone layouts.

Each fixture should specify the expected accepted/rejected candidates and relative ranking constraints.

### End-to-end evaluation

Maintain a small set of representative images with reviewed outputs. Tests should verify schema validity, finite coordinates, repeatable candidate order, and successful rendering without requiring exact pixel-for-pixel image equality.

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Diamonds are hidden or absent | Automatic homography fails | Geometry confidence gate and manual corner fallback |
| Balls occupy very few pixels | Misclassification increases | Input-quality warning, optional crop guidance, larger model comparison |
| Dataset is small | Detector overfits | Pretrained weights, conservative augmentation, held-out tests, later personal images |
| Related video frames leak across splits | Metrics become misleading | Perceptual grouping and manifest-based split |
| Angled views shift the apparent ball centre | Table positions are biased | Compare centre and view-angle correction on matched views |
| Stretch resizing distorts geometry | Angles and distances become wrong | Preserve aspect ratio and track all resize transforms |
| Different tables have different pocket mouths | Ranking assumptions drift | Configurable pocket radius and later table-specific calibration |
| Dot false positives corrupt line fitting | Homography becomes unstable | RANSAC, spacing constraints, reprojection checks, confidence gate |
| Ball detections overlap | Duplicate state objects | Class-agnostic NMS and game-count constraints |
| Broadcast images differ from personal photos | Domain shift | Collect a small independent personal evaluation set before claiming generalization |

## 9. Milestones and indicative workload

| Milestone | Result | Indicative effort |
|---|---|---:|
| 0 | Project skeleton and reproducible environment | 0.5-1 day |
| 1 | Audited, repaired, leakage-aware dataset | 1-2 days |
| 2 | Trained and evaluated detector | 1-3 days plus training time |
| 3 | Automatic and manual homography | 2-4 days |
| 4 | Stable normalized table-state JSON | 1-2 days |
| 5 | Direct-shot candidate generator | 2-3 days |
| 6 | Explainable ranking and scratch heuristic | 2-3 days |
| 7 | CLI, rendering, tests, and documentation | 2-3 days |

The expected personal-project schedule is approximately three to five focused weekends. Detector training can proceed while geometry unit tests and synthetic shot tests are developed.

## 10. Definition of done for MVP

The MVP is complete when:

- A fresh environment can be installed from documented commands.
- Dataset preparation runs deterministically and passes all audit checks.
- The detector has recorded validation and held-out test metrics.
- Automatic dot-based homography succeeds on most unobstructed evaluation images.
- Manual calibration handles automatic failures.
- Every accepted ball has a valid normalized position.
- Direct legal shots are generated and blocked shots are excluded in unit tests.
- The CLI returns the top three suggestions with score explanations.
- Annotated and top-down outputs are produced for representative images.
- Known limitations are documented without claiming exact physical simulation.

## 11. Future extensions

After the MVP is stable, possible extensions are:

- Bank and kick shots using mirrored table geometry.
- Temporal tracking and state smoothing for video.
- Cue direction estimation.
- More accurate collision and scratch simulation.
- Calibration from pockets or table segmentation when diamonds are unavailable.
- Fine-tuning with personal phone-camera images.
- A Streamlit or web interface.
- Multi-turn strategic search.
- Comparison against the paper's RL agents or a learned ranking model.

## 12. Immediate next actions

1. Create the project skeleton and dependency configuration.
2. Implement the read-only dataset audit.
3. Visually verify and repair the `18t` dot annotation in the generated processed copy.
4. Generate the detection and geometry evaluation splits with a manifest.
5. Train a YOLO11n baseline.
6. Implement dot-line diagnostics before attempting complete homography automation.

## References

- Pix2Pockets project: <https://pix2pockets.compute.dtu.dk/>
- Paper: <https://arxiv.org/abs/2504.12045>
- Dataset: <https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o>
- Reference implementation: <https://github.com/viktorseba/pix2pockets>

