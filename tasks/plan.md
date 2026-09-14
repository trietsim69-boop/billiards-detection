# Implementation Plan: Finish Phase 1 and Continue Phase 2

Date: 2026-09-14
Status: proposed implementation plan; no implementation performed by this planning task.
Implementation repository: `D:\billiards project`, currently `main` at `9315675`.
Task checklist: [todo.md](todo.md).

These planning files live in `D:\billiards project\tasks`. All source,
configuration, test, and output paths below are relative to `D:\billiards project`
unless explicitly stated otherwise. No existing plan was overwritten.

## Objective

Close the remaining Phase 1 prototype gates, then produce a traceable normalized
table state on clear, fully visible broadcast images. First prove the table
transform and ball projection using reviewed geometry and labelled balls. Then
connect the selected detector and evaluate automatic geometry, with an explicit
manual four-corner fallback. Shot ranking and further detector training are
outside this implementation plan.

## Verified starting point

- Git history remains at `9315675`; the progress tracker has uncommitted updates.
  Existing tracked bytecode changes also remain in the worktree.
- An authorized read outside the sandbox on September 14 found all expected
  image/label pairs: train 155, validation 20, test 20, geometry evaluation 52.
  Git reported no dataset deletions. Earlier apparent deletions were caused by
  restricted access; no dataset restoration is needed on this evidence. Counts
  establish presence, not a new content-integrity audit.
- Baseline training, the 960-pixel retrain, validation reports, and crop
  experiments already exist. The original 960-pixel full-frame detector remains
  the current candidate; crop supplements remain experimental.
- Phase 1.6 still lacks the complete shared detection contract and conservative
  ball filtering. A geometry-specific detector/confidence selection is pending.
- Four-rail fitting already exists. The reported 19/20 validation successes at
  confidence 0.05 belong to the 640-pixel model and Dot-label corner agreement.
  They do not establish playable-bed or ball-projection accuracy.
- The September 12 check passed 14 lightweight tests. The complete suite failed
  during collection with PyTorch `WinError 1455` while loading `cufft64_11.dll`.
  This is historical evidence; Task 1 checks the current environment again.
- `TableLocalizationResult.from_corners` already creates a pixel-sized rectangle
  transform. It does not yet provide the explicit normalized 2:1 table-state
  contract. Its automatic color-based corner proposal is experimental.

## Architecture decisions

1. **Finish a bounded detector milestone.** Use existing checkpoints. Evaluate a
   finite set of configurations and select one; do not restart training merely
   because the optional crop experiment stopped after three epochs.
2. **One detection contract.** Keep class, confidence, original-image box and
   centre, stable prediction ID, and provenance. Define "raw" as the candidates
   exposed by the configured Ultralytics prediction call, including its existing
   NMS, before project-specific filtering. Preserve all removed candidates and
   reasons. Geometry consumes this contract.
3. **Keep experiments separate from the default.** Compare the original 640 and
   960 full-frame models and, where saved evidence permits, original-model
   Dot-only crops. Consider the saved retrained Dot supplements only if their
   geometry benefit justifies the additional checkpoint and inference cost.
   Full-frame ball predictions remain the initial path.
4. **Distinguish physical planes.** Rail sights provide rail-line and ordering
   evidence. A sight-derived homography is not automatically a playable-bed
   homography. Use reviewed inner-cushion corners to prove bed projection and
   validate automatic proposals against that reference. This follows the newer
   two-plane rule in `RAIL_FIRST_DETECTION_PLAN.md`.
5. **Build a usable manual path early.** Four reviewed playable-bed corners should
   yield a normalized state before automatic geometry is integrated. Reuse the
   current corner validation where appropriate, with an explicitly named new
   normalized transform so existing rail-strip consumers retain their contract.
6. **Freeze groups before geometry tuning.** Keep all views of one situation in
   the same development or holdout group. Use validation and geometry development
   data for selection. Run geometry holdout after code and thresholds are frozen;
   keep the detector test split sealed until final detector choices are fixed.
7. **Failures remain visible.** Report rejection rates, wrong accepted results,
   and errors over matched image subsets. Never improve the reported score by
   silently removing difficult images or changing acceptance thresholds.

## Ordered work

### Close Phase 1

- [ ] Task 1: Verify the current runtime.
- [ ] Task 2: Export repeatable raw detections for one image.
- [ ] Task 3: Export traceable filtered detections.
- [ ] Checkpoint A: Detector output works from image to JSON.
- [ ] Task 4: Select and document the detector operating point.

Phase 1 is closed only when Tasks 1-4 pass the existing prototype gates. Its
completion does not claim production accuracy or sealed-test performance.

### Establish the Phase 2 reference

- [ ] Task 5: Freeze the grouped geometry manifest.
- [ ] Checkpoint B: Phase 1 signed off and geometry groups fixed.
- [ ] Task 6: Capture reviewed playable-bed geometry.
- [ ] Task 7: Produce a normalized table state from four corners.
- [ ] Task 8: Establish the labelled-ball geometry baseline.
- [ ] Checkpoint C: Ground-truth geometry and manual calibration work.

### Integrate automatic inputs

- [ ] Task 9: Verify canonical sight correspondence.
- [ ] Task 10: Evaluate automatic playable-bed geometry.
- [ ] Task 11: Connect the detector to table-state output.
- [ ] Checkpoint D: Automatic success and manual fallback are distinguishable.
- [ ] Task 12: Run frozen evaluation and record the phase verdict.
- [ ] Checkpoint E: Phase 2 evidence and remaining limitations are reviewable.

## Dependencies

```text
1 -> 2 -> 3 -> 4                 Phase 1 completion
1 -> 5 -> 6 -> 7 -> 8            Reviewed/manual geometry
          |         |
          +-------> 9 -> 10      Sight correspondence and bed validation
4 + 7 + 8 + 9 + 10 -> 11 -> 12   Integration and frozen evaluation
```

Task 5 can proceed independently of detector tuning after Task 1. Task 9 uses
labelled dots and can proceed alongside the reviewed/manual geometry path once
groups and annotation conventions are fixed. Work sharing the detection schema
or geometry CLI must be sequenced. This plan does not require multiple agents.

## Definition of Done

For every task, meet its acceptance criteria, run the focused tests, exercise the
actual input/output path, and preserve existing behavior outside the requested
change. Record configuration and relevant data/model hashes for experiments.
Update interface documentation when behavior changes. Keep existing user edits
and dataset contents intact. Do not stage tracked bytecode or unrelated files.

There is no documented package build or lint command in the inspected repository.
Use focused tests and CLI runs, with `python -m compileall -q src scripts` as a
syntax check where appropriate; do not report a nonexistent build as passing.
At each checkpoint run the complete existing test suite if the runtime permits.
Report environment failures explicitly rather than treating skipped tests as passes.

No numerical real-image Phase 2 accuracy target is invented here. Task 8 measures
annotation repeatability and freezes explicit residual, bed-corner, coverage, and
ball-layout acceptance limits on development data before Task 10 selection and
Task 12 holdout evaluation. Synthetic numerical correctness has fixed tolerances
in the task checklist. Acceptance limits cannot be relaxed after seeing holdout.

## Risks and handling

| Risk | Consequence | Planned handling |
|---|---|---|
| Windows memory pressure | PyTorch or inference cannot start | Verify current imports first; use bounded batch-one execution and diagnose if failure repeats |
| Restricted dataset access | False missing-file reports | Check under an authorized account before proposing any restore or relocation |
| Small validation set | Overfitting configurations | Compare a bounded candidate set and freeze before grouped holdout |
| Rail sights lie above the bed | Plausible warp with biased ball positions | Separate transforms and use reviewed inner-cushion geometry |
| Perspective makes the apparent long side misleading | Rotated or stretched table state | Record physical long-side orientation and reject unresolved correspondence |
| Ball box centres are above bed contact points | View-dependent ball displacement | Measure using matched views; retain as a documented MVP approximation if it meets the frozen bar |
| Existing locator performs poorly | Automatic Phase 2 gate remains open | Report measured failures; manual output can work while automatic completion remains explicitly blocked |

## Decisions to resolve through the tasks

- Determine whether `WinError 1455` still occurs; no system-setting change is
  required by this planning task.
- Select the detector/confidence pair using the existing fitter and identical
  validation images; do not assume higher detector mAP means better geometry.
- Verify the 25 situation identifiers before choosing deterministic group counts.
- Determine real-image geometry limits from reviewed development evidence.
- If the current automatic bed locator cannot meet those limits with a bounded
  correction, write a separate targeted follow-up. Do not silently expand this
  plan into a new learned detector or relabel manual success as automatic success.

## Immediate next session

Execute Tasks 1-3, ending with one image producing both raw and filtered JSON plus
an overlay. The next checkpoint is a dependable detector interface. Task 4 then
closes Phase 1; Tasks 5-8 establish the first usable Phase 2 table-state path.

## Source documents

- `TABLE_DETECTION_PLAN.md`: Phase 1.6, prototype gates, and Phase 2 checkpoints.
- `TABLE_DETECTION_PROGRESS.md`: September 12 detector and geometry evidence.
- `RAIL_FIRST_DETECTION_PLAN.md`: grouped evaluation and two-plane rule.
- `outputs/experiments/crop-finetune-v1/REPORT.md`: crop experiment limitations.
