# Rail-First Detection Plan

Status: initial rail-first path regressed; crop-aware fine-tuning evaluated after three completed epochs
Branch: `rail-first-detection`  
Scope: detect coarse table geometry before rail sights, normalize four rail regions, improve sight-centre recovery, and preserve a trustworthy bed-plane mapping

### 2026-09-07 experimental checkpoint

The first all-validation comparison is recorded in
`outputs/geometry/rail-first/validation-960-20260907/REPORT.md`.
It uses existing 960-pixel weights without retraining. A provisional image-only
cloth locator supplies proposals on 14/20 images. At YOLO confidence 0.25 and
8px centre tolerance at 960 scale, full-frame YOLO matches 324/359 labelled dots
with 12 false positives; crop YOLO matches 61 with 5 false positives; the crop
hybrid matches 92 with 47 false positives. The initial implementation is not
promoted. Unit tests alone did not establish detection improvement.

This is an early diagnostic, not completion of phases A-G below. Reviewed bed
annotations, grouped geometry manifests, complete projective strip normalization,
robust rail roles/lattice fitting, and geometry accuracy evaluation remain open.
Test and geometry holdout images have not been used in this experiment.

### 2026-09-07 crop-aware fine-tuning follow-up

`outputs/experiments/crop-finetune-v1/REPORT.md` records the follow-up. Four
overlapping, aspect-preserving rectangular crops supplement full-frame YOLO;
they do not depend on the provisional cloth locator or rail lattice. Training
uses independent copies of 155 original training frames plus 620 labelled crops,
retaining all five classes. Three epochs completed, not the planned twelve:
one at 960 pixels and two resumed at 640 after memory failures. Both saved
checkpoints were evaluated at the same 960-pixel inference resolution.

On all 20 validation images, at fixed full/crop confidences 0.25/0.50 and 8px
centre tolerance, the original detector finds 324/359 dots with 12 false
positives. Original-model crops improve this to 341 with 14 false positives.
Keeping the original full-frame model and adding epoch-3 crop dots gives 343
with 13 false positives: only two additional matches from retraining itself.
Replacing the full-frame model with epoch 3 instead gives 343 matches with 31
false positives. The original default checkpoint therefore remains unchanged.
Supplementary crops should be Dot-only if integrated: adding crop ball classes
also increased several ball-class false-positive counts.

The unchanged rail fitter accepts 14/20 original, 16/20 original-plus-crop, and
17/20 original-plus-epoch-3-crop results. This is not reviewed bed-plane accuracy
or sealed generalization evidence. Eleven of the sixteen remaining misses in
the last variant are in one distant-table validation image. The next training
experiment should maintain a consistent high-resolution scale on hardware with
sufficient memory and target small/distant training examples. Phases A-G remain
open; neither the normalized rail-first path nor a new production model has
been promoted.

## 1. Goal

Build and evaluate a rail-first detection path over the repository's full permitted
datasets. The approach must determine whether coarse table localization and
rail-region normalization improve downstream sight detection and table geometry
relative to the existing full-frame YOLO pipeline.

The intended flow is:

```text
image
  -> coarse cloth / inner-cushion localization
  -> ordered bed quadrilateral + confidence
  -> four normalized rail search regions
  -> learned and/or classical sight candidates
  -> per-rail spacing and shape validation
  -> accepted sight centres in source-image coordinates
  -> existing four-line geometry verification
  -> separate bed-plane mapping for ball projection
```

This work is dataset-first. No individual image is the baseline. External phone
photos may be included later as qualitative or out-of-domain examples, but they
must not determine thresholds or architecture.

## 2. Questions the branch must answer

1. Can the inner cloth/cushion quadrilateral be localized reliably before sights
   are known?
2. Does normalizing each rail region improve sight-centre recall, especially on
   small and far rails?
3. Is crop-only YOLO sufficient, or does a bright low-chroma proposal path add
   useful candidates?
4. Can the expected 6/3 rail-wise spacing pattern remove classical false
   positives without deleting real occluded sights?
5. Does the hybrid improve rail recovery and homography evidence without
   accepting confidently wrong geometry?
6. Which transform belongs to the bed plane, and which transforms are only for
   searching the raised rail caps?

## 3. Dataset inventory and split policy

The processed corpus contains 247 images:

| Split | Images | Role in this branch |
|---|---:|---|
| `train` | 155 | Algorithm development; learned table-localizer training if later justified |
| `valid` | 20 | Candidate thresholds, model selection, and A/B/C/D comparison |
| `test` | 20 | Sealed until the implementation and thresholds are frozen |
| `geometry_eval` | 52 | Matched views from 25 situations; grouped geometry development and holdout evaluation |

Rules:

- Treat `data/processed/pix2pockets_v3` as immutable input.
- Never select thresholds from the detector `test` split.
- Split `geometry_eval` by situation, never by image, so alternate views of one
  table state cannot cross development and holdout groups.
- Create one checked-in manifest for the grouped geometry split and reuse it in
  every experiment.
- Use `train` for implementation diagnostics, `valid` for selection, the geometry
  development situations for homography work, and the geometry holdout situations
  for the final pre-test comparison.
- Report metrics over complete splits and per rail. Galleries and named examples
  are diagnostic evidence only.
- Do not use external phone images to tune in-domain thresholds. A useful OOD
  evaluation requires a collection, not one photograph.

### Geometry grouping checkpoint

Before geometry tuning:

1. Recover or create situation IDs for all 52 `geometry_eval` images.
2. Verify that the 52 views represent the documented 25 situations.
3. Create a deterministic grouped development/holdout manifest.
4. Record image counts, situation counts, table-area distribution, and view count
   per situation.
5. Freeze the manifest before selecting rail-first geometry thresholds.

## 4. Dataset-wide baseline

The branch begins by measuring the current approach across permitted datasets,
not by reproducing one failure image.

Use the current 960-pixel checkpoint and run one inference pass per image at a low
minimum confidence. Reuse the raw detections for every higher confidence threshold.

Baseline datasets:

- all 20 `valid` images;
- all geometry development images;
- all geometry holdout images for a no-tuning comparison after the baseline
  settings are frozen.

Baseline measurements:

- sight-centre precision, recall, and F1 at normalized centre-distance tolerances;
- sight recall per image and per rail;
- recall by table area, sight size, and near/far rail where orientation permits;
- raw and filtered candidate counts;
- number of rails with enough correct support;
- four-line recovery and structural validity;
- corner agreement against a geometry oracle;
- reprojection residual when canonical correspondences are available;
- runtime and peak memory.

The existing 640-pixel reports remain historical evidence. The new comparison
must use the same 960-pixel raw predictions, data manifest, matching rules, and
geometry acceptance rules for every rail-first variant.

## 5. Geometry annotations

No separate duplicate image dataset is required. Add a versioned annotation file
that references the existing immutable images.

Proposed location:

```text
data/annotations/table_geometry_v1.jsonl
```

Each record should support both fully visible and cropped tables:

```json
{
  "image": "data/processed/pix2pockets_v3/valid/images/example.jpg",
  "split": "valid",
  "situation_id": null,
  "image_width": 1920,
  "image_height": 1080,
  "inner_cushion_lines": [
    {"side": 0, "p1": [0.0, 0.0], "p2": [0.0, 0.0], "visible": true},
    {"side": 1, "p1": [0.0, 0.0], "p2": [0.0, 0.0], "visible": true},
    {"side": 2, "p1": [0.0, 0.0], "p2": [0.0, 0.0], "visible": true},
    {"side": 3, "p1": [0.0, 0.0], "p2": [0.0, 0.0], "visible": true}
  ],
  "bed_corners_xy": [
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0]
  ],
  "quality": "reviewed",
  "notes": ""
}
```

Annotation conventions:

- Annotate the inner cushion nose / playable-bed boundary, not the outside wooden
  table edge and not the row of sights.
- Store coordinates in original-image pixels and store image dimensions.
- Order sides and derived corners clockwise using one documented image-space
  convention.
- Collect two points on each visible inner-cushion line; derive infinite line
  equations and intersections in code.
- Permit derived corner coordinates outside the image for cropped but recoverable
  tables.
- Mark genuinely unobservable sides explicitly rather than guessing them.
- Keep hand-authored observations separate from derived lines, corners,
  homographies, and confidence scores.
- Review a sample twice to estimate annotation consistency before labelling the
  complete selected set.

Initial annotation scope:

1. Annotate all 20 validation images.
2. Annotate all grouped geometry development and holdout images.
3. Add training annotations only if a learned table locator becomes necessary.
4. Leave detector test geometry annotations sealed or create them only after the
   method and thresholds are frozen.

Existing Dot labels remain the sight-centre oracle. The new file supplies the
bed/inner-cushion oracle that the current YOLO labels do not contain.

## 6. Module design

### 6.1 Coarse table localization module

Proposed file:

```text
src/billiards/table_localization.py
```

Small interface:

```python
localize_table(image, config=None) -> TableLocalizationResult
```

The implementation may use adaptive color estimation, morphology, connected
components, long-edge extraction, robust line clustering, quadrilateral scoring,
and 2:1 plausibility checks. Callers should not need to understand those details.

The result must expose:

- validity and explicit warnings;
- four ordered inner-cushion lines;
- four ordered bed corners, including finite off-image intersections;
- quadrilateral area and plausibility measurements;
- a coarse bed-plane homography and its inverse when valid;
- confidence/evidence fields needed for diagnostics;
- no sight detections and no hidden fallback to Dot labels.

### 6.2 Rail-region module

Proposed file:

```text
src/billiards/rail_regions.py
```

Small interface:

```python
extract_rail_regions(image, table, config=None) -> tuple[RailRegion, ...]
```

Each returned rail region must include:

- a normalized strip image;
- the source-image polygon used for sampling;
- forward and inverse point transforms;
- expected long/short-side role when it can be established safely;
- valid-pixel coverage for cropped strips;
- warnings when too little of a rail is observable.

The transform round trip must be testable without running a detector.

### 6.3 Rail-sight module

Proposed file:

```text
src/billiards/rail_sights.py
```

Small interface:

```python
detect_rail_sights(image, table, learned_candidates=(), config=None) \
    -> SightDetectionResult
```

The implementation should:

- map full-frame learned candidates into probable rail regions;
- optionally run the existing detector on normalized rail crops through the CLI;
- create bright low-chroma/top-hat connected-component proposals;
- merge spatial duplicates while preserving candidate provenance;
- score normalized size, contrast, solidity, shape, and distance to the sight row;
- fit a one-dimensional equal-spacing lattice per rail;
- allow missing sights, occlusion, and a possible name-plate position;
- map accepted centres back to original-image coordinates;
- retain rejected candidates and reasons for diagnostics.

YOLO model loading remains outside this pure geometry/candidate module. Scripts
pass predictions in, which keeps tests fast and deterministic.

### 6.4 Existing geometry module

Keep `src/billiards/geometry.py` input-agnostic. Accepted source-image sight
centres from the new pipeline are passed to the unchanged four-line fitter as an
independent structural verification path.

Do not silently replace or weaken current warnings merely because a coarse table
proposal exists.

## 7. Two-plane rule

The implementation must name and preserve different transforms:

- `bed_homography`: derived from the inner cushion / playable-bed geometry and
  intended for table-plane reasoning;
- `rail_strip_transform`: used to normalize one raised rail-cap search region;
- any future sight-template transform: derived from rail-cap points and never
  silently treated as the bed transform.

Detected ball-box centres are also not automatically bed contact points. Ball
projection accuracy remains a separate later checkpoint.

## 8. Implementation phases and checkpoints

### Phase A — inventory, manifests, and baseline

- [ ] Verify all 247 processed image/label pairs and record inaccessible files.
- [ ] Create and freeze the grouped `geometry_eval` manifest.
- [ ] Run the 960-pixel full-frame detector once on complete permitted splits.
- [ ] Generate dataset, image, rail, size, and geometry baseline reports.
- [ ] Save configuration, raw predictions, CSV/JSON metrics, and contact sheets.

Completion condition: every later variant can be compared against one immutable,
dataset-wide baseline without rerunning YOLO differently.

### Phase B — annotation tooling and bed oracle

- [ ] Define and validate the JSONL schema.
- [ ] Add a simple annotation/review command for four inner-cushion lines.
- [ ] Annotate validation and grouped geometry images.
- [ ] Render every annotation and inspect a contact sheet.
- [ ] Measure repeat-annotation line/corner variation on a sample.

Completion condition: reviewed bed geometry exists for every selected development
and holdout image, with no split leakage.

### Phase C — coarse table localization

- [ ] Start with image-adaptive cloth color rather than a fixed blue threshold.
- [ ] Close holes from balls, pockets, shadows, and overlays.
- [ ] Fit long inner-cushion boundaries robustly.
- [ ] Score candidate quadrilaterals by support, convexity, area, corner margin,
  opposite-line families, and rectified aspect plausibility.
- [ ] Return invalid with diagnostics when evidence is insufficient.
- [ ] Compare predicted lines and corners with the new annotation oracle.

Completion condition: localization quality is reported across full selected splits;
no hand-picked image determines the parameters.

### Phase D — rail extraction

- [ ] Construct four source-image rail-cap bands with generous plane-offset margins.
- [ ] Normalize strips at a resolution that preserves far-rail sights.
- [ ] Save strip mosaics for every evaluation image.
- [ ] Verify source-to-strip-to-source point round trips.
- [ ] Record strip coverage for cropped or occluded rails.

Completion condition: each valid coarse table produces four traceable rail regions
whose transforms meet the numerical round-trip test.

### Phase E — candidate variants

Implement four comparable variants:

| Variant | Candidate source | Geometry prior |
|---|---|---|
| A | Existing full-frame 960 YOLO | Existing four-line fitter |
| B | YOLO on normalized rail crops | Coarse rail membership |
| C | Bright/low-chroma and local-contrast blobs | Rail regions and normalized size |
| D | Union of B and C | Rail regions, candidate provenance, and lattice |

- [ ] Use identical source images and matching rules.
- [ ] Do not select a different table locator per candidate variant.
- [ ] Preserve raw candidates before geometric filtering.
- [ ] Measure candidate recall before lattice selection and precision afterward.

Completion condition: the contribution of crop normalization, classical proposals,
and lattice filtering can be distinguished rather than bundled into one result.

### Phase F — rail-wise lattice and final verification

- [ ] Assign candidates to one of four probable rails.
- [ ] Fit six positions on long rails and three on short rails in normalized
  one-dimensional coordinates.
- [ ] Allow missing observations and reject unsupported extra positions.
- [ ] Map accepted centres back to source coordinates.
- [ ] Run the existing `fit_rails` implementation on those centres.
- [ ] Calculate bed-corner, rail-line, lattice, and reprojection residuals.
- [ ] Reject invalid geometry rather than returning a forced homography.

Completion condition: every accepted result has traceable candidates, rail
assignments, residuals, and acceptance reasons.

### Phase G — selection and sealed evaluation

- [ ] Select thresholds and the winning variant using `valid` and geometry
  development data only.
- [ ] Freeze configuration and code before geometry holdout evaluation.
- [ ] Compare the frozen method on grouped geometry holdout situations.
- [ ] Inspect all false accepts and false rejects.
- [ ] Run the detector test split once only after the method is frozen.
- [ ] Record whether the rail-first branch should replace, augment, or remain a
  fallback to the existing pipeline.

Completion condition: the decision is based on split-wide evidence and includes
failure/rejection rates, not only successful-image averages.

## 9. Metrics and reporting

### Table localization

- supported line length and mask evidence per side;
- inner-cushion line distance error;
- corner mean, median, and p95 error in pixels and as an image-diagonal ratio;
- valid quadrilateral rate;
- false-accept and false-reject rates;
- table area and rectified-aspect distributions.

### Sight candidates

- centre precision, recall, and F1 at fixed normalized tolerances;
- per-rail recall and rails with minimum support;
- recall by source box size, table area, and rail perspective;
- raw candidates per image and accepted candidates per image;
- candidate provenance: full-frame YOLO, crop YOLO, classical, or union;
- lattice residual and number of inferred missing positions.

### Geometry

- four-line recovery and structural validity;
- corner agreement with the Dot-label oracle and bed annotation oracle, reported
  separately because they represent different physical lines/planes;
- homography inlier count and symmetric reprojection error;
- matched-view projected-position error on grouped geometry situations;
- explicit rejection rate and confidently wrong acceptance count.

### Runtime

- mean and p95 time per image;
- detector and classical-stage time separately;
- peak GPU and system memory where available.

Every run writes:

```text
outputs/geometry/rail-first/<run-name>/
    config.yaml
    manifest.csv
    summary.json
    per_image.csv
    per_rail.csv
    candidates.jsonl
    overlays/
    rail_strips/
    contact_sheet.jpg
    REPORT.md
```

## 10. Tests

### Pure geometry tests

- line normalization and intersection ordering;
- convex and non-convex quadrilaterals;
- vertical, horizontal, and steep perspective rails;
- intersections outside the image;
- source/strip transform round trips;
- invalid image sizes and degenerate lines.

### Localization tests

- synthetic cloth quadrilaterals with balls, pockets, shadows, and occlusions;
- multiple large blue/green regions;
- non-table images that must fail closed;
- partial tables with insufficient evidence;
- deterministic results for identical input and configuration.

### Sight tests

- white balls inside the bed are not rail sights;
- bright background objects outside rail bands are rejected;
- round and diamond-shaped sight candidates are allowed;
- missing, duplicated, and occluded lattice positions;
- 3-position and 6-position rails;
- crop/full-frame candidate deduplication;
- accepted strip centres map accurately back to source coordinates.

### Integration tests

- stored small fixtures exercise the complete pipeline without loading YOLO;
- predicted candidates are injected through the public interface;
- result JSON round-trips and preserves warnings and provenance;
- existing `fit_rails` tests continue to pass unchanged.

## 11. Failure policy and fallback

The system must fail closed when any of these occur:

- fewer than four supported inner-cushion lines;
- degenerate, non-convex, implausibly small, or implausibly large bed geometry;
- excessive off-image extrapolation without visible line support;
- too little valid rail-strip coverage;
- insufficient per-rail sight support;
- high lattice or reprojection residual;
- disagreement between coarse geometry and accepted sight geometry beyond a
  configured validation threshold.

Failure output must retain overlays, intermediate candidates, warnings, and scores.
The planned manual four-corner path remains the final fallback; this branch must
not convert uncertainty into a plausible-looking but unsupported 2D table.

## 12. Decision rules

Rail-first is promoted only if, on the same frozen evaluation inputs, it:

1. improves or preserves dataset-wide sight-centre recall;
2. improves or preserves four-rail recovery and geometry validity;
3. reduces geometry error or expands valid coverage on difficult small-table
   images;
4. does not increase confidently wrong accepted geometry;
5. retains explicit rejection and manual fallback paths; and
6. has acceptable runtime and memory for the target machine.

If crop-only YOLO provides the gain, stop before adding classical complexity. If
classical candidates improve recall but lattice filtering cannot control false
accepts, retain them as diagnostics rather than production inputs. If coarse table
localization is the dominant failure, collect/train a learned bed segmentation or
corner model using the same annotation schema instead of tuning increasingly
brittle color constants.

## 13. Expected repository changes

```text
RAIL_FIRST_DETECTION_PLAN.md
data/annotations/table_geometry_v1.jsonl
configs/rail_first.yaml
src/billiards/table_localization.py
src/billiards/rail_regions.py
src/billiards/rail_sights.py
scripts/annotate_table_geometry.py
scripts/evaluate_rail_first.py
tests/test_table_localization.py
tests/test_rail_regions.py
tests/test_rail_sights.py
tests/fixtures/rail_first/
```

Implementation must not restore, delete, or stage unrelated dataset/worktree
changes. Generated evaluation artifacts remain under `outputs/`.

## 14. Related evidence

- `research/rail_first_sight_detection.md` records the primary-source study
  comparison and the rationale for a hybrid rail-first design.
- `TABLE_DETECTION_PLAN.md` remains the broader detector and geometry plan.
- `TABLE_DETECTION_PROGRESS.md` records the current 640/960 detector and
  four-line-fitting evidence.
