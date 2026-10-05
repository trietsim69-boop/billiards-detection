# Phase 1 Closure and Phase 2 Task Checklist

Date: 2026-09-14; Task 4 and Phase 2 (Tasks 5-11) replanned 2026-10-05. Plan: [plan.md](plan.md).
All implementation paths and commands below refer to `D:\billiards project`.
Use its `.venv\Scripts\python.exe`; `python` below abbreviates that interpreter.
Commands for proposed scripts/tests become executable when their task is implemented.
Generated reports, JSON, and overlays go under unique `outputs/` run directories;
source-file scope estimates exclude generated per-image artifacts.

## Task 1: Verify the current runtime

**Description:** Establish that the saved detector can run today. Dataset presence
was rechecked during planning: all 247 image/label pairs exist and Git reports no
dataset deletions. Inspect access under the correct account if that discrepancy
recurs. Reproduce the earlier memory error only if it still occurs.

**Acceptance criteria:**

- [ ] Record interpreter, package versions, checkpoint presence, and data access without modifying the corpus.
- [ ] PyTorch imports and one validation image runs through the original 960-pixel checkpoint at batch one; save the command and result.
- [ ] The complete suite passes, or a specific reproducible environment blocker is recorded and resolved before claiming runtime readiness.

**Verification:** `python -m pytest -q`; one local YOLO prediction with saving disabled or redirected into a unique output directory. Inspect actual class outputs. No package build is configured.

**Dependencies:** None.
**Files likely touched:** `REAL_IMAGE_TESTING.md`; generated `outputs/verification/<run>/runtime.md`.
**Estimated scope:** Small, 1 documentation file; no planned model training.

## Task 2: Export repeatable raw detections

**Description:** Add a shared detection representation and a CLI that converts
one image into raw detection JSON and an overlay. Reuse the existing prediction
conversion and crop structures where practical, avoiding a broad analyzer refactor.

**Acceptance criteria:**

- [ ] Each detection exposes a stable ID, class ID/name, confidence, original-image box/centre, and source; output records image dimensions, model identity/hash, and inference settings.
- [ ] The adapter explicitly defines upstream NMS and letterbox coordinate behavior; all five classes and empty results serialize successfully without NaN/Infinity.
- [ ] Replaying saved predictions produces identical detection content and IDs; repeated image inference is numerically consistent within documented tolerances.

**Verification:** `python -m pytest -q tests/test_detection_contract.py`; `python scripts/detect_image.py --help`; run the CLI on a validation image and inspect JSON coordinates against its overlay. Check syntax for the touched modules.

**Dependencies:** Task 1.
**Files likely touched:** `src/billiards/detection.py`, `scripts/detect_image.py`, `tests/test_detection_contract.py`.
**Estimated scope:** Medium, 3 files.

## Task 3: Export traceable filtered detections

**Description:** Complete the same image-to-JSON path with the domain filtering
already specified in Phase 1.6. Keep raw evidence available for every decision.

**Acceptance criteria:**

- [ ] Apply configurable confidence thresholds, conservative class-agnostic ball overlap suppression, and caps of one Cue, one Black, seven Solid, seven Striped, and sixteen total balls.
- [ ] Preserve raw detections; every retained or removed detection refers to its original ID, with deterministic removal reasons and warnings for conflicting alternatives.
- [ ] Keep Dot candidates separate from ball rules; do not manufacture eighteen dots or apply an unvalidated size/count truncation. JSON and overlays expose raw and filtered states.

**Verification:** `python -m pytest -q tests/test_detection_contract.py tests/test_detection_filtering.py`; verify conflicting ball classes, excess counts, empty input, and low-confidence dots; inspect representative small-table and clear-table validation outputs.

**Dependencies:** Task 2.
**Files likely touched:** `src/billiards/detection.py`, `scripts/detect_image.py`, `tests/test_detection_filtering.py`, `REAL_IMAGE_TESTING.md`.
**Estimated scope:** Medium, 4 files.

## Checkpoint A: Image-to-detections path

- [ ] Tasks 1-3 pass; one command emits raw/filtered JSON and a reviewable overlay.
- [ ] The full existing test suite passes; runtime limitations are recorded if still unresolved.
- [ ] Output schema and removed-candidate diagnostics are ready for review.

## Task 4: Select the detector operating point

**Description:** Close Phase 1 with a bounded comparison using all 20 validation
images and unchanged rail-fitting gates. Compare the 960 model (repo default) with
the 1280 Colab model; first copy its weights from Google Drive
(`MyDrive/8ballpool/outputs/yolo11n-1280`) into `outputs/detection/`. Score dot
centres at 8 px (960) and 10.7 px (1280), and re-sweep the rail Dot confidence.
The 640 and crop artifacts were removed; their recorded numbers stand and are not
re-run. Select by geometry correctness and coverage as well as detection quality
and runtime.

**Acceptance criteria:**

- [ ] Every candidate uses the same images, matching rules, and rail gates; report per-class ball metrics, Dot metrics, corner agreement, false accepts, rejection rates, and runtime, including matched-subset comparisons.
- [ ] Save one explicitly selected configuration with model hash, resolution, ball thresholds, Dot threshold, NMS, and crop policy; load it through the detection CLI. Do not copy the 640-specific 0.05 threshold blindly.
- [ ] Run the selected contract on representative validation images, preserve reproducible outputs, and mark Phase 1.6/prototype gates complete only when supported. Test and geometry holdout remain unused for selection.

**Verification:** `python -m pytest -q`; `python scripts/sweep_rail_confidence.py --help`; execute the bounded validation comparison in a unique output directory and inspect each rejected or incorrectly accepted rail fit. No training run is required.

**Dependencies:** Tasks 1-3.
**Files likely touched:** `scripts/sweep_rail_confidence.py`, `configs/detection.yaml`, `scripts/detect_image.py`, `tests/test_rail_confidence_sweep.py`, `TABLE_DETECTION_PROGRESS.md`.
**Estimated scope:** Medium, up to 5 files.

## Task 5: Freeze geometry dev/holdout groups

**Description:** Assign the 25 geometry situations to dev or holdout once,
deterministically, and commit the list. `split_manifest.csv` already maps every
image to `situation-NN`, so no new manifest file is needed.

**Acceptance criteria:**

- [ ] `configs/geometry.yaml` gains a sorted `holdout_situations` list, the seed, and the one-line command that produced it (13 dev / 12 holdout unless plan Open Question 4 says otherwise).
- [ ] A test reads the manifest and config: 52 geometry images, 25 situations, every image in exactly one group, no situation in both, and every situation with its top view.
- [ ] No later task opens holdout images before Task 11.

**Verification:** `.venv\Scripts\pytest.exe -q tests/test_geometry_split.py`. No model run.

**Dependencies:** None.
**Files likely touched:** `configs/geometry.yaml`, `tests/test_geometry_split.py`.
**Estimated scope:** XS, 2 files.

## Task 6: Labelled dots → sight homography with numbered overlay

**Description:** First vertical slice. A labelled image goes through `fit_rails`,
sight-to-template correspondence and `cv2.findHomography`, and comes out as a
numbered sight overlay, a warped top-down image and JSON. Reuse the label
loaders, image discovery and rail rendering in `scripts/fit_table_rails.py`.

**Acceptance criteria:**

- [ ] `src/billiards/table_state.py` provides an 18-point sight template with the rail offset (`sight_offset_in: 3.6875`, `bed_width_in: 50` in `configs/geometry.yaml`). Its correspondence step requires 6/3/6/3 rail support, orders each rail's inliers along the rail and maps clockwise to clockwise. Any other input returns a reason such as `sight_pattern:6/3/5/3` or `invalid_rail_fit`, never a homography.
- [ ] Synthetic test: template projected through a known perspective homography, shuffled, plus two off-rail outliers. The recovered `H` puts every sight within 1e-5 units and round-trips image points within 1e-3 px. Rotating the input 180° gives the 180°-equivalent state; one missing sight gives a failure reason.
- [ ] `scripts/detect_table_state.py --mode labels --source <image|dir> --output outputs/geometry/<run>` writes the overlay, warp and JSON per image (`method: automatic_sights`, `H` image→table, sight residuals, failure reason). Run it on the 20 validation images and all dev views, and report full-pattern coverage on dev. If most dev views fail, raise partial-pattern matching before Task 8.

**Verification:** `.venv\Scripts\pytest.exe -q tests/test_table_state.py tests/test_geometry.py`; `python -m compileall -q src scripts`; inspect every overlay and warp: straight rails, side pockets at `x = 1`, no mirroring.

**Dependencies:** Task 5 for dev images (validation images can be used earlier).
**Files likely touched:** `src/billiards/table_state.py`, `scripts/detect_table_state.py`, `tests/test_table_state.py`, `configs/geometry.yaml`.
**Estimated scope:** Medium, 4 files.

## Task 7: Balls → normalized table state, plus manual corners

**Description:** Complete the image-to-state path. Labelled balls project into
table coordinates, and four supplied inner-cushion corners give the manual path
through the same code.

**Acceptance criteria:**

- [ ] Ball box centres (Black, Cue, Solid, Striped) project through `H` into `x ∈ [0,2], y ∈ [0,1]`. JSON keeps class, source index, `image_xy` and `table_xy`. Balls more than one ball radius outside the bed are flagged with a reason, never clamped. A 1000×500 top-down diagram shows pockets and class-coloured balls.
- [ ] `--corners x1 y1 x2 y2 x3 y3 x4 y4` (inner-cushion corners, clockwise, first edge a long rail) maps the bed to `(0,0),(2,0),(2,1),(0,1)` with `method: manual_corners`. Nonfinite, duplicate, crossing or degenerate corners are rejected with a reason. A failed automatic fit without `--corners` returns its failure; nothing is substituted silently.
- [ ] Synthetic test: balls placed in table coordinates, projected into the image and back, round-trip within 1e-3 px, and the JSON has no NaN/Infinity. `REAL_IMAGE_TESTING.md` documents both methods and their parallax approximation. If plan Open Question 1 is approved, update the two-plane bullet in `CLAUDE.md`.

**Verification:** `.venv\Scripts\pytest.exe -q tests/test_table_state.py`; run one automatic and one manual example; compare each top-down diagram with its source image.

**Dependencies:** Task 6.
**Files likely touched:** `src/billiards/table_state.py`, `scripts/detect_table_state.py`, `tests/test_table_state.py`, `REAL_IMAGE_TESTING.md`, `CLAUDE.md`.
**Estimated scope:** Medium, 5 files.

## Checkpoint B: Labelled image → table state

- [ ] Tasks 5-7 pass; one command turns a labelled image, or an image plus four corners, into table-state JSON, overlay, warp and top-down diagram.
- [ ] The full suite and `compileall` pass.
- [ ] Overlays and the JSON schema reviewed with the human before measuring accuracy.

## Task 8: Matched-view accuracy on dev; freeze limits

**Description:** Measure label-mode geometry the way Pix2Pockets did: every
non-top dev view against its situation's top view. Add one absolute check for
errors that matched views cannot see.

**Acceptance criteria:**

- [ ] `scripts/evaluate_table_state.py` runs label mode on all dev images. It aligns each non-top view to its top view (0°/180°, minimum error) and matches balls per class by minimum total distance (permutations, ≤7 per class, no scipy). It reports per-view and overall median/p90/max error in units and cm (1 unit = 127 cm), split by `a`/`f` view, plus coverage and every rejection with its reason.
- [ ] Absolute check: sight-derived bed corners against human-entered inner-cushion corners on 5 dev top views (`data/annotations/bed_corners_dev.csv`). A systematic scale error means the config (`sight_offset_in`, `bed_width_in`) is wrong, not the code.
- [ ] Acceptance limits go into `configs/geometry.yaml` from dev evidence only (proposed: p90 ≤ one ball radius, plan Open Question 2). Record whether Task 9 is needed: yes if the bar is missed or angled views err along the viewing direction.

**Verification:** `.venv\Scripts\pytest.exe -q tests/test_geometry_evaluation.py` (matching and 180° alignment on synthetic fixtures); run the evaluator; inspect the five worst views.

**Dependencies:** Task 7; the human supplies the 20 corner points.
**Files likely touched:** `scripts/evaluate_table_state.py`, `tests/test_geometry_evaluation.py`, `configs/geometry.yaml`, `data/annotations/bed_corners_dev.csv`, `TABLE_DETECTION_PROGRESS.md`.
**Estimated scope:** Medium, 5 files.

## Task 9 (conditional): Project balls onto the ball-centre plane

**Description:** Build this only if Task 8 calls for it. Recover the focal length
and camera pose from `H_sight` (principal point at the image centre, square
pixels). Then map image points onto the plane `z = R`, where ball centres are,
instead of the sight plane. Fit the sight height above the bed (one scalar) on
dev matched views.

**Acceptance criteria:**

- [ ] Synthetic camera test: with known intrinsics and pose, points at `z = R` are recovered within 1e-4 units. Near-top-down views, where the focal length is ill-conditioned, fall back to `H_sight` with a warning.
- [ ] The manual-corners path uses the same correction (bed plane → ball plane).
- [ ] On the same dev views, matched-view error improves on Task 8. The fitted sight height and before/after numbers are recorded, and the Task 8 limits stay unchanged.

**Verification:** `.venv\Scripts\pytest.exe -q tests/test_table_state.py`; rerun the Task 8 evaluator.

**Dependencies:** Task 8.
**Files likely touched:** `src/billiards/table_state.py`, `tests/test_table_state.py`, `configs/geometry.yaml`.
**Estimated scope:** Small, 3 files.

## Checkpoint C: Label-mode geometry measured

- [ ] Dev matched-view error, coverage and the absolute check are recorded; limits are frozen in config.
- [ ] Task 9 is either done or explicitly skipped with its reason.
- [ ] Reviewed with the human before any YOLO input.

## Task 10: YOLO detections → table state

**Description:** Replace labels with the detector selected in Task 4, read
through the Task 3 filtered-detection contract.

**Acceptance criteria:**

- [ ] `--mode yolo` uses the Task 4 model and Dot confidence plus Task 3 filtered balls, and the JSON records the model SHA256 and settings. A rail with more inliers than sight slots keeps its highest-confidence dots (marked `# ponytail:` heuristic, ceiling noted); otherwise it is rejected.
- [ ] The evaluator runs YOLO mode on the Task 8 dev images and reports error, coverage, false accepts (accepted geometry over the limit) and the failing stage (detector, rail fit, sight pattern, projection) beside the label results.
- [ ] A rejected dev image rerun with `--corners` produces a `manual_corners` state.

**Verification:** `.venv\Scripts\pytest.exe -q`; CLI runs on one accepted image, one rejected image, and the rejected image with corners.

**Dependencies:** Tasks 3, 4, 8 (and 9 if built).
**Files likely touched:** `scripts/detect_table_state.py`, `scripts/evaluate_table_state.py`, `tests/test_table_state.py`, `REAL_IMAGE_TESTING.md`.
**Estimated scope:** Medium, 4 files.

## Checkpoint D: Automatic inputs

- [ ] Automatic success, structured rejection and manual recovery are each shown on dev images.
- [ ] Label and YOLO results on identical images are side by side; the full suite passes.

## Task 11: Frozen holdout run and verdict

**Description:** Freeze everything, run the holdout situations once, and record
the Phase 2 verdict.

**Acceptance criteria:**

- [ ] Commit hash, weights SHA256 and configs are written to `outputs/geometry/<run>/` before the first holdout inference; no code or threshold changes afterwards.
- [ ] The report covers the full holdout denominator: label and YOLO error against the frozen limits, coverage, false accepts, manual recoveries and runtime.
- [ ] `TABLE_DETECTION_PROGRESS.md`, `README.md` and the `CLAUDE.md` status are updated. A missed bar becomes a named follow-up, not a retune.

**Verification:** `.venv\Scripts\pytest.exe -q`; `python -m compileall -q src scripts`; report counts match the config's holdout list.

**Dependencies:** Task 10.
**Files likely touched:** `TABLE_DETECTION_PROGRESS.md`, `README.md`, `CLAUDE.md`.
**Estimated scope:** Small, 3 files.

## Checkpoint E: Completion evidence

- [ ] Every task's acceptance criteria and the Definition of Done in `plan.md` are met.
- [ ] Phase 1 and Phase 2 status, uncertainty and follow-ups are backed by saved evidence.
- [ ] Ready for human review; no merge implied.

## Planning verification

- [x] Each task has at most three acceptance criteria, verification, dependencies and a file scope (≤5 files).
- [x] Phase 2 reuses `fit_rails`, the label/YOLO loaders and the existing split manifest; nothing depends on removed code.
- [x] Phase 1 Tasks 1-3 were kept; Task 4 now targets the 960/1280 comparison.
- [ ] Human has reviewed the plan and answered its open questions.
