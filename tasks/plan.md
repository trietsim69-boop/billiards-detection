# Implementation Plan: Finish Phase 1 and Build Phase 2

Date: 2026-10-05 (replans Phase 2 of the 2026-09-14 plan; Phase 1 Tasks 1-3 unchanged)
Status: proposed; awaiting human review. Repository `main` at `5a04781`.
Task checklist: [todo.md](todo.md).

## Overview

Phase 1 ends with a traceable detection JSON and a selected detector/Dot
threshold. Phase 2 turns one image into a normalized 2:1 table state: rail sights
→ sight-to-template correspondence → homography → ball positions in
`x ∈ [0,2], y ∈ [0,1]`, with a manual four-corner fallback. It is built on
labelled dots and balls first, measured on matched views, and only then fed YOLO
detections. Shot ranking is out of scope.

## Verified starting point

- `fit_rails` (`src/billiards/geometry.py`) returns four rails in clockwise
  corner order with their inlier indices. Labelled-dot fits show 6/3/6/3 support
  on 19/20 validation images (one has 6/3/5/3).
- `scripts/fit_table_rails.py` already loads labelled or YOLO dots into
  `PointObservation`s (`--mode labels|yolo`); reuse its loaders and renderers.
- `split_manifest.csv` maps every geometry image to `situation-NN`: 25
  situations, each with one top view (`t`) and 27 angled/front views (`a`/`f`;
  situations 7 and 21 have both).
- No table-state, homography or correspondence code exists. The old plan's
  `table_localization.py` and `TableLocalizationResult.from_corners` were removed
  in `11b8b8b`; nothing below depends on them.
- `scipy` is not installed. `match_centers` in `analyze_dot_centers.py` is a
  maximum-cardinality matcher, not a minimum-cost one.

Physical constants (WPA equipment spec): sight centres sit 3 11/16 in (93.7 mm)
outward from the cushion nose; cushion nose height is 63.5% of ball diameter
(≈1.43 in); ball radius 1.125 in. A 9-ft bed is 100 × 50 in, so one normalized
unit = 50 in = 127 cm and the sight offset is d ≈ 0.074 units.

## Architecture decisions

1. **The sight template includes the rail offset.** Long-rail sights at
   `x = 0.25, 0.50, 0.75, 1.25, 1.50, 1.75` on `y = −d` and `y = 1 + d`;
   short-rail sights at `y = 0.25, 0.50, 0.75` on `x = −d` and `x = 2 + d`.
   Putting sights on the bed edge (the TECHNICAL_PLAN hypothesis) shrinks the
   table by ≈2d ≈ 15%. `sight_offset_in` and `bed_width_in` live in
   `configs/geometry.yaml`.
2. **Balls project through the sight homography in the automatic path.** A ball
   box centre images the ball *centre*, 1.125 in above the bed. Sights sit on the
   rail top, roughly 1.5–2 in above the bed (to be measured). The sight plane is
   therefore closer to the ball-centre plane than the bed is, so `H_sight`
   carries less parallax than a true bed homography would. The exact fix
   (camera pose from `H_sight`, then the plane `z = R`) is Task 9, built only if
   Task 8 shows it is needed. This keeps the two-plane rule (a sight transform is
   not the bed homography). It drops "ball projection uses reviewed
   inner-cushion corners" for the automatic path; see Open Question 1.
3. **Full sight pattern only in v1.** 6/3/6/3 rail support or a structured
   rejection that points to manual corners. Partial-pattern slot matching
   (cross-ratio) waits for Task 6's coverage numbers.
4. **The 180° ambiguity is a table symmetry, not a bug.** Clockwise image order
   mapped to clockwise template order fixes handedness. The remaining 0°/180°
   choice yields an equivalent game state. Evaluation aligns it to the top view
   by minimum error.
5. **Measure with matched views, not a new annotation campaign.** The 27
   non-top views are compared with their situation's top view using labelled
   balls. Matched views cannot see errors common to every view (wrong offset,
   wrong table size), so a small absolute check uses human-entered bed corners
   on 5 dev top views. This replaces the old annotation tool, pilot and
   repeat-annotation tasks.
6. **Manual fallback = four inner-cushion corners on the CLI** (`--corners`).
   Bed plane, same projection code, `method: manual_corners`. No silent fallback
   and no click UI yet.
7. **Labels first, YOLO last.** Tasks 5–9 need no detector or torch and can run
   alongside Phase 1 Tasks 1–4. Only Task 10 waits for Phase 1.
8. **Kept from the previous plan:** freeze holdout before tuning; every image
   stays in the denominator; failures and false accepts are reported; the
   detector test split stays sealed.

## Task list

### Phase 1 close (unchanged except Task 4)
- [ ] Task 1: Verify the current runtime
- [ ] Task 2: Export repeatable raw detections
- [ ] Task 3: Export traceable filtered detections
- [ ] Checkpoint A: image → detection JSON
- [ ] Task 4: Select the detector operating point (960 vs 1280, Dot threshold)

### Phase 2a: labelled image → table state
- [ ] Task 5: Freeze geometry dev/holdout groups
- [ ] Task 6: Labelled dots → sight homography with numbered overlay
- [ ] Task 7: Balls → normalized table state, plus manual corners
- [ ] Checkpoint B: one command, labelled image → table-state JSON and overlays

### Phase 2b: accuracy on dev
- [ ] Task 8: Matched-view accuracy on dev; freeze limits
- [ ] Task 9 (conditional): Project balls onto the ball-centre plane
- [ ] Checkpoint C: label-mode geometry measured, limits frozen

### Phase 2c: automatic inputs and verdict
- [ ] Task 10: YOLO detections → table state
- [ ] Checkpoint D: automatic success, rejection and manual recovery on dev
- [ ] Task 11: Frozen holdout run and verdict
- [ ] Checkpoint E: Phase 2 verdict reviewable

## Dependencies

```text
Phase 1:  1 -> 2 -> 3 -> 4
Phase 2:  5 -> 6 -> 7 -> 8 -> (9 if Task 8 misses its bar)
          3 + 4 + 8 (+9) -> 10 -> 11
```

Tasks 6–7 can use validation images (labelled dots, no matched views) before
Task 5 is done; dev geometry images wait for Task 5.

## Definition of Done

Each task meets its acceptance criteria, its focused tests pass, the real CLI
path runs on real images, and overlays are inspected. No build or lint exists:
use `.venv\Scripts\pytest.exe -q` and `python -m compileall -q src scripts`.
Runs write to fresh `outputs/` directories with config, weights SHA256 and
commit hash. Run the full suite at every checkpoint.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Angled views hide near-rail sights, so few views have the full 6/3/6/3 pattern | High: automatic coverage collapses | Task 6 reports dev coverage first; if most dev views fail, pull partial-pattern matching forward before Task 8 |
| Non-9-ft tables (bar boxes) have a different `d` | Medium: uniform scale error that matched views cannot see | Task 8 absolute check; table size is a config value |
| The top view is not exact ground truth (it has small parallax too) | Low | Report matched-view error as consistency; the absolute check is separate |
| Extra collinear YOLO dots at conf 0.05 break the 6/3 count | Medium | Task 10 keeps the top-confidence dots per rail (marked heuristic) and measures false accepts |
| Pose recovery is ill-conditioned near top-down views | Low: parallax is tiny there anyway | Task 9 falls back to `H_sight` with a warning |
| Labelled dots missing on some images | Low | Rejected with a reason and counted, not dropped |

## Open questions

1. **Approve decision 2?** Automatic ball projection goes through sights.
   Reviewed corners are used only for the manual fallback and the absolute
   check. If approved, Task 7 updates the two-plane bullet in `CLAUDE.md`.
2. **Ball-error bar:** proposed p90 matched-view error ≤ one ball radius
   (≈2.9 cm, 0.022 units), the scale at which shot feasibility starts to
   change. Accept or set a bar before Task 8 sees dev numbers.
3. **Absolute check:** who reads the 4 inner-cushion corner pixels on 5 dev top
   views (20 points)? An agent cannot click.
4. **Split size:** proposed 13 dev / 12 holdout situations.

## Out of scope

Shot ranking (Phase 3), detector retraining, partial-pattern matching (unless
Task 6 forces it), a click UI for corners, geometry confidence calibration,
anchor heuristics between box centre and top edge (superseded by Task 9).

## Source documents

`TABLE_DETECTION_PLAN.md` (Phase 2 checkpoints 2.1–2.9),
`TABLE_DETECTION_PROGRESS.md`, `RAIL_FIRST_DETECTION_PLAN.md` (two-plane rule),
Pix2Pockets (arXiv 2504.12045) for the matched-view metric (0.4 cm mean).
