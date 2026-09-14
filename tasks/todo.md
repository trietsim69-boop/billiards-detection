# Phase 1 Closure and Phase 2 Task Checklist

Date: 2026-09-14. Plan: [plan.md](plan.md).
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
images and unchanged rail-fitting gates. Reuse compatible saved predictions.
Compare the original 640/960 models and justified Dot-only crop candidates; select
by geometry correctness and coverage as well as detection quality and runtime.

**Acceptance criteria:**

- [ ] Every candidate uses the same images, matching rules, and rail gates; report per-class ball metrics, Dot metrics, corner agreement, false accepts, rejection rates, and runtime, including matched-subset comparisons.
- [ ] Save one explicitly selected configuration with model hash, resolution, ball thresholds, Dot threshold, NMS, and crop policy; load it through the detection CLI. Do not copy the 640-specific 0.05 threshold blindly.
- [ ] Run the selected contract on representative validation images, preserve reproducible outputs, and mark Phase 1.6/prototype gates complete only when supported. Test and geometry holdout remain unused for selection.

**Verification:** `python -m pytest -q`; `python scripts/sweep_rail_confidence.py --help`; execute the bounded validation comparison in a unique output directory and inspect each rejected or incorrectly accepted rail fit. No training run is required.

**Dependencies:** Tasks 1-3.
**Files likely touched:** `scripts/sweep_rail_confidence.py`, `configs/detection.yaml`, `scripts/detect_image.py`, `tests/test_rail_confidence_sweep.py`, `TABLE_DETECTION_PROGRESS.md`.
**Estimated scope:** Medium, up to 5 files.

## Task 5: Freeze the grouped geometry manifest

**Description:** Make a deterministic development/holdout assignment for the 52
geometry images, keeping all views of each of the documented 25 situations together.

**Acceptance criteria:**

- [ ] Every image has a verified situation ID and view label, appears exactly once, and stays grouped with its other views; mismatches in the documented situation count are investigated before freezing.
- [ ] Save the chosen seed, group counts, split assignment, and source identifiers/hash; keep detector test examples out of geometry development.
- [ ] Mark a clear/full-pattern development subset using recorded visibility criteria; preserve the full manifest and report exclusions without silently dropping evaluation cases.

**Verification:** `python -m pytest -q tests/test_geometry_split.py`; regenerate in memory twice and compare assignments; review the manifest counts and verify no situation crosses groups. No model run is needed.

**Dependencies:** Task 1.
**Files likely touched:** `scripts/prepare_geometry_split.py`, `configs/datasets/geometry_groups.csv`, `tests/test_geometry_split.py`.
**Estimated scope:** Medium, 3 files.

## Checkpoint B: Phase boundary

- [ ] Tasks 4-5 pass; the selected detector satisfies the existing Phase 1 prototype gates.
- [ ] Group assignment is frozen before geometry-specific tuning.
- [ ] Detector selection evidence and geometry manifest are ready for review.

## Task 6: Capture reviewed playable-bed geometry

**Description:** Add a small annotation/import workflow for four inner-cushion
lines or corners, with overlays and explicit orientation. Start with a pilot of
clear development images to establish the convention before annotating more.

**Acceptance criteria:**

- [ ] Store image dimensions/ID, group, corner order, physical long-side role, visibility, provenance, and reviewed status; reference immutable images rather than duplicate them.
- [ ] Annotate the cushion-nose/playable-bed boundary, distinguish it from the sight row, and reject invalid or genuinely unobservable geometry instead of guessing it.
- [ ] Inspect every pilot overlay and independently repeat a subset of annotations to measure variation before applying the convention to the larger reference set.

**Verification:** `python -m pytest -q tests/test_geometry_annotations.py`; `python scripts/annotate_table_geometry.py --help`; import or capture one annotation, reload it, and inspect its rendered corner/line labels.

**Dependencies:** Task 5.
**Files likely touched:** `src/billiards/geometry_annotations.py`, `scripts/annotate_table_geometry.py`, `data/annotations/table_geometry_v1.jsonl`, `tests/test_geometry_annotations.py`.
**Estimated scope:** Medium, 4 files; bounded pilot only.

## Task 7: Produce a normalized table state from four corners

**Description:** Deliver the first complete Phase 2 path: an image, reviewed
playable-bed corners, and labelled balls produce a 2:1 table state, a warped view,
and a top-down ball diagram. The same explicit corner input becomes the manual fallback.

**Acceptance criteria:**

- [ ] Map ordered physical bed corners to `(0,0),(2,0),(2,1),(0,1)` with named forward/inverse normalized transforms, without changing the current pixel-sized rail-strip contract.
- [ ] Project ball anchors while retaining class, source ID, and original coordinates; output JSON, source overlay, and top-down diagram with the calibration method. Ball-box-centre projection is identified as an approximation.
- [ ] Reject nonfinite, crossing, duplicate, degenerate, or unresolved-orientation inputs; validate projected points and preserve failure reasons without clamping them into a plausible table.

**Verification:** `python -m pytest -q tests/test_table_state.py`; assert synthetic normalized corner error below `1e-5` and source-coordinate round-trip error below `1e-3` px on well-conditioned fixtures; run `python scripts/detect_table_state.py --help` and a manual-corner example with labelled balls.

**Dependencies:** Task 6; reuse Task 2 representation where useful.
**Files likely touched:** `src/billiards/table_state.py`, `scripts/detect_table_state.py`, `tests/test_table_state.py`, `REAL_IMAGE_TESTING.md`.
**Estimated scope:** Medium, 4 files.

## Task 8: Establish the labelled-ball geometry baseline

**Description:** Extend reviewed annotations across validation and geometry
development images using the pilot convention. Compare projected labelled balls
across matched views, then freeze real-image acceptance limits before selecting
the automatic geometry path.

**Acceptance criteria:**

- [ ] Every selected development image has reviewed geometry or an explicit unobservable status; repeat-annotation variation and all exclusions are reported. Process annotation batches separately if the full set exceeds one session.
- [ ] Report bed-corner error, projection residuals, matched-view ball-position error, orientation consistency, and usable coverage; use explicit matches for same-class balls and expose ambiguous correspondences.
- [ ] Record justified numerical acceptance limits and the target domain in geometry configuration using development evidence only. Do not use the four input corners' own zero reprojection error as independent validation.

**Verification:** `python -m pytest -q tests/test_geometry_evaluation.py tests/test_table_state.py`; run `python scripts/evaluate_table_state.py --help` and its development evaluation; inspect every overlay and matched-view layout in the report.

**Dependencies:** Tasks 5-7.
**Files likely touched:** `scripts/evaluate_table_state.py`, `tests/test_geometry_evaluation.py`, `configs/geometry.yaml`, `data/annotations/table_geometry_v1.jsonl`.
**Estimated scope:** Medium, 4 files; annotation batches are bounded review sessions.

## Checkpoint C: Reviewed/manual geometry

- [ ] Tasks 6-8 pass; manual corners and labelled balls produce a correct, repeatable normalized layout.
- [ ] Synthetic tests pass and real-image acceptance limits are documented before automatic selection.
- [ ] Corner conventions, annotation variation, and ball-anchor limitations are ready for review.

## Task 9: Verify canonical sight correspondence

**Description:** Add a diagnostic that orders labelled rail sights and relates
them to the canonical 6/3 pattern. Keep the sight geometry distinct from the
playable-bed geometry already established by reviewed corners.

**Acceptance criteria:**

- [ ] Number and visualize the labelled sight correspondences on source/template views; account for side-pocket gaps and the verified long/short-side orientation.
- [ ] Reject ambiguous, insufficient, or inconsistent ordering; shuffled input produces the same valid assignment. Handle the initial clear/full-pattern scope explicitly.
- [ ] Export sight correspondences, residuals, and any sight transform with its physical-plane label; never feed that transform into ball projection as if it were the bed transform.

**Verification:** `python -m pytest -q tests/test_sight_correspondence.py tests/test_geometry.py`; run `python scripts/check_sight_correspondence.py --help` and render all eligible labelled development cases. Verify orientation on steep-perspective fixtures.

**Dependencies:** Tasks 5-6; Task 8 before any new real-image threshold selection.
**Files likely touched:** `src/billiards/sight_correspondence.py`, `scripts/check_sight_correspondence.py`, `tests/test_sight_correspondence.py`.
**Estimated scope:** Medium, 3 files.

## Task 10: Evaluate automatic playable-bed geometry

**Description:** Measure the existing automatic table locator against reviewed
inner-cushion geometry and use sight evidence as a separate consistency check.
Attempt only corrections attributable to measured failures within this task's scope.

**Acceptance criteria:**

- [ ] Report predicted bed-corner/line errors, orientation mistakes, rejected cases, and falsely accepted geometry across complete development splits, using Task 8 limits.
- [ ] Accepted automatic geometry has explicit evidence for bed boundaries and long-side orientation; color-mask agreement or sight-line structure alone cannot certify it.
- [ ] Promote only if the frozen development bar is met. Otherwise record the automatic gate as open and name the next targeted change; the usable manual path does not imply automatic completion.

**Verification:** `python -m pytest -q tests/test_table_localization.py tests/test_geometry_evaluation.py tests/test_sight_correspondence.py`; run the development evaluation, inspect every false accept, and verify insufficient geometry returns a failure reason.

**Dependencies:** Tasks 8-9.
**Files likely touched:** `src/billiards/table_localization.py`, `scripts/evaluate_table_state.py`, `tests/test_table_localization.py`, `tests/test_geometry_evaluation.py`, `configs/geometry.yaml`.
**Estimated scope:** Medium, up to 5 files; new learned localization training requires a separate follow-up plan.

## Task 11: Connect the detector to table-state output

**Description:** Extend the working table-state CLI to consume Phase 1 detections,
check rails, choose validated automatic bed geometry when available, and accept
manual corners when automatic geometry fails.

**Acceptance criteria:**

- [ ] A single input image produces detections, geometry diagnostics, normalized balls, and both visual outputs with model/config identity and coordinate systems recorded.
- [ ] Detector, sight, bed, and projection failures remain distinguishable; failed automatic calibration requests corners or returns a structured failure, and supplied manual corners are labelled as manual.
- [ ] Compare predicted inputs with the labelled baseline on identical development images; record accuracy, failure coverage, and runtime without changing frozen thresholds to mask failures.

**Verification:** `python -m pytest -q tests/test_table_state_pipeline.py tests/test_table_state.py tests/test_detection_filtering.py`; CLI checks for one valid automatic case, one rejected case, and that rejected case with manual corners. If Task 10 remains open, verify manual integration and leave automatic acceptance unchecked.

**Dependencies:** Tasks 4, 7-10; automatic success requires Task 10 to pass.
**Files likely touched:** `scripts/detect_table_state.py`, `src/billiards/table_state.py`, `tests/test_table_state_pipeline.py`, `REAL_IMAGE_TESTING.md`.
**Estimated scope:** Medium, 4 files.

## Checkpoint D: Integrated state detection

- [ ] Tasks 9-11 meet their acceptance criteria; the full suite and CLI paths pass.
- [ ] Automatic success, explicit rejection, and manual recovery are independently demonstrated.
- [ ] Any unpassed automatic gate remains open; the integration report is ready for review.

## Task 12: Run frozen evaluation and record the phase verdict

**Description:** Freeze code, weights, settings, and group manifest before final
geometry evaluation. Complete reviewed holdout annotations using the established
convention without model-guided corrections, then report the frozen pipeline.

**Acceptance criteria:**

- [ ] Record a versioned run manifest before holdout inference. Holdout annotation review follows the fixed convention and does not tune thresholds; detector test is used only after detector choices are also frozen.
- [ ] Report the complete holdout denominator, coverage, false accepts/rejects, bed and ball errors, manual recovery, and runtime against the predetermined limits; manual and automatic results remain separate.
- [ ] Update the progress and main plan with passed/open gates and exact artifact locations. A failed gate prevents a Phase 2 completion claim and produces a bounded follow-up rather than retuning against the same holdout.

**Verification:** `python -m pytest -q`; `python -m compileall -q src scripts`; run the frozen evaluator and CLI examples, inspect all false accepts, and verify reported counts against the manifest. Review only the intended source/document diff; preserve existing user edits and omit bytecode.

**Dependencies:** Tasks 1-11 and their checkpoints.
**Files likely touched:** `scripts/evaluate_table_state.py`, `configs/geometry_evaluation.yaml`, `data/annotations/table_geometry_v1.jsonl`, `TABLE_DETECTION_PROGRESS.md`, `TABLE_DETECTION_PLAN.md`.
**Estimated scope:** Medium, up to 5 files; holdout annotation review is batched without algorithm changes.

## Checkpoint E: Completion evidence

- [ ] Every task's acceptance criteria and the shared Definition of Done are met.
- [ ] Phase 1 status, Phase 2 status, uncertainty, and any follow-up are supported by saved evidence.
- [ ] Changes and results are ready for human review; no merge or deployment is implied.

## Planning verification

- [x] Each task has at most three acceptance criteria, verification, dependencies, and a file scope.
- [x] Existing code, experimental results, and the newer two-plane requirement informed the plan.
- [x] Dataset presence was verified outside the restricted sandbox; restoration was removed from the proposed work.
- [x] No existing incomplete plan was overwritten; no implementation was performed.
- [x] Checkpoints separate the major deliverables.
- [ ] Human has reviewed the proposed implementation plan.
