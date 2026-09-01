# Rail-first sight detection and table rectification

_Prepared 2026-09-01. Scope: primary papers, author code, official equipment specifications, and methods transferable to this repository. “Sight” is used for the white dot/diamond markers on a pool-table rail._

## Recommendation in one paragraph

Yes, table/rail detection before sight detection is worth testing. The best design is not a one-way replacement for the current detector, but a two-pass bootstrap: estimate a **coarse** table quadrilateral from the cloth/cushion boundary, rectify or separately normalize four narrow rail bands, detect bright low-chroma sight candidates only inside those bands, enforce the known rail-wise spacing pattern, and then refine the final homography from the validated sights. This should improve tiny-sight recall by removing most of the image, normalizing perspective and scale, and making geometric false-positive rejection much stronger. The coarse cloth homography should not be trusted as the final sight homography because the sights are on the raised rail cap rather than the cloth plane.

## What the closest studies actually do

### 1. Pix2Pockets: dots first, then rails and homography

The closest published system is [Pix2Pockets (Schiøtt, Petersen, and Papadopoulos, SCIA 2025)](https://arxiv.org/html/2504.12045), with the [authors' code](https://github.com/viktorseba/pix2pockets). It:

1. detects balls and white rail sights with a fine-tuned YOLOv5 model;
2. repeatedly applies RANSAC to the detected sight centres to recover four lines;
3. checks the line intersections, groups and orders the sights, and uses the 3-sight short-side versus 6-sight long-side pattern to orient the table;
4. estimates a homography from up to 18 sights plus four inferred corners rather than only four points.

The paper reports post-processed sight precision/recall of 90.8%/90.8%, sight AP50 of 89.3%, and a mean cross-view projected ball-position shift of 0.4 cm. Its baseline is particularly relevant to the “white dot” idea: adaptive thresholding followed by Hough circles produced class-agnostic AP50:95 of only 0.21 when dots were included, versus 0.67 for the learned detector. In other words, whiteness/circularity alone was not competitive on in-the-wild broadcast images. The exact RANSAC and homography implementation is visible in the authors' [`mapping.py`](https://raw.githubusercontent.com/viktorseba/pix2pockets/main/auxillary/mapping.py).

Pix2Pockets therefore validates the current repository's use of low-confidence sight candidates plus geometric consensus. It does **not** test table-first sight detection, so the expected benefit of rectifying before detecting sights remains a hypothesis requiring an ablation here.

### 2. Billiards studies that localize the table before downstream detection

Several primary billiards studies use the order the user proposed:

| Primary source | Table localization | What happens after localization | Transferable lesson |
|---|---|---|---|
| [Legg et al., “Intelligent Filtering by Semantic Importance for Single-View 3D Reconstruction From Snooker Video,” ICIP 2011](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2011/papers/1569407575.pdf) | RGB→HSL, threshold the baize hue, connected components, Hough boundary lines, four intersections | Warp the table to a top-down rectangle, then constrain object filtering to known table bounds | A table mask and four boundaries can be a useful first-stage calibration even with an angled tripod view. |
| [Uchiyama et al., “AR Supporting System for Pool Games Using a Camera-Mounted Handheld Display,” 2008](https://onlinelibrary.wiley.com/doi/10.1155/2008/357270) | Compare pixel color with a learned table color, extract the mask contour, cluster line segments into four groups in rho-theta space, intersect them | Compute a frame-by-frame homography and then estimate ball positions | Table color should be estimated from the scene, not assumed globally; multiple boundary segments can be clustered before corner estimation. |
| [Park and Park, “Intelligent Carom Billiards Assistive System…,” 2022](https://journals.sagepub.com/doi/10.1177/17298806221118865) | Lens calibration, HSV table-bed threshold, contour extraction, Douglas–Peucker polygon approximation, select the largest plausible quadrilateral | Use its four corners for an image-to-object homography and restrict ball detection to the table ROI | Undistort first; a quadrilateral table-bed proposal is an effective ROI and coarse metric map in a controlled setup. |
| [Sousa et al., “PoolLiveAid,” 2013](https://sapientia.ualg.pt/bitstream/10400.1/3386/1/wscg2013.pdf) | Temporal averaging, Canny edges, Hough lines, group candidate boundaries, intersect four lines; manual adjustment is allowed | Apply a perspective transform when the camera is off-centre, then run ball/cue/trajectory processing | Boundary calibration can be computed once for a fixed camera and should retain a manual correction path. |

These studies are controlled-camera systems and mostly use green felt, so their fixed thresholds cannot simply be copied into this broadcast-image project. Their pipeline ordering and geometric constraints are the transferable part.

### 3. Official sight geometry is stronger than “white” appearance

The [World Pool-Billiard Association equipment specification](https://www.wpapool.com/wp-content/uploads/2024/01/RECOMMENDED-EQUIPMENT-SPECIFICATIONS.pdf) states that a regulation table has 18 sights, or 17 plus a name plate. On a 9-foot table, adjacent sights are 12.5 inches apart and their centres are 3 11/16 inches from the cushion nose. Sights may be round, 7/16–1/2 inch in diameter, or diamond-shaped, approximately 1×7/16 to 1 1/4×5/8 inch. The specification does not require them to be white.

Consequences for this repository:

- High lightness and low chroma are useful candidate features for the present dataset, but should not define the universal class.
- A detector must allow both compact round and elongated diamond shapes.
- Equal rail-wise spacing, expected count, distance from the cushion, and one possible name-plate substitution are more reliable priors than color alone.
- A hard “exactly 18 white dots” rule would wrongly reject occlusions, non-white sights, and compliant 17-plus-nameplate tables.

## Why rail-first should help here

The local evidence in [`TABLE_DETECTION_PROGRESS.md`](../TABLE_DETECTION_PROGRESS.md) shows that all labelled sights are below 32×32 pixels and distant/small-table frames are the principal failure mode. Moving from 640 to 960 input improved Dot recall from 0.640 to 0.786, which is evidence that usable pixel scale matters. The 640 model also showed that low-confidence candidates plus RANSAC can recover structurally valid rails on 19/20 validation images, even when ordinary detector F1 would choose a much higher confidence threshold.

A coarse table-first stage should provide four benefits:

1. **Scale normalization.** Each rail can be resampled to a fixed strip height so a distant sight is no longer represented by only a few pixels.
2. **Perspective normalization.** Far-rail sights become closer in size to near-rail sights, allowing one candidate-size range and one local detector.
3. **Search-space reduction.** White balls, score graphics, clothing, lights, and logos away from the rail bands disappear before candidate generation.
4. **Stronger structure.** After rectification, each rail is a one-dimensional sequence with three or six nearly equally spaced positions, rather than an unordered set of white blobs in a full frame.

These are geometric reasons to expect improved candidate recall and precision. No located paper reports a controlled before/after sight-detection ablation, so the improvement should be measured rather than stated as a known result.

## Important geometry caveat: there are two planes

A planar homography is valid for points on the plane used to estimate it. The table bed/cloth is one plane; sights are mounted flush on the raised rail cap, a parallel but different plane. [Uchiyama et al.](https://onlinelibrary.wiley.com/doi/10.1155/2008/357270) explicitly formulates the homography from four coplanar table corners, and the WPA specification places sights on the rail cap.

At a steep viewing angle, applying a cloth-plane homography to rail-cap sights can therefore create a systematic positional offset. This does not invalidate table-first processing, but changes its role:

- use the cloth homography as a **coarse ROI and scale normalizer**;
- include generous rail-band margins;
- preferably rectify each rail cap as its own quadrilateral strip if outer and inner rail edges are visible;
- estimate/refine the **final** table mapping from sight-template correspondences and rail lines, with residual checks.

The same caveat matters if cloth corners are used for final ball projection: ball contact points belong to the bed plane, while detected ball-box centres do not. Keep the final ball-position model separate from sight-strip rectification.

## Proposed two-pass pipeline

### Pass A — coarse table and rail localization

1. Undistort when camera intrinsics are known. Park and Park calibrate before table extraction; this prevents curved boundaries from corrupting line fits. For unknown broadcast cameras, retain a no-undistortion path and let downstream residuals expose problematic frames.
2. Estimate a felt/table-bed mask. Prefer a learned segmentation mask or an image-adaptive hue model over fixed green thresholds because the WPA permits yellow-green, blue-green, and electric-blue cloth and this dataset contains broadcast variation.
3. Close small holes caused by balls, shadows, cushion breaks, and pockets; fit long boundary segments robustly rather than trusting a raw four-vertex contour.
4. Generate candidate convex quadrilaterals, enforcing plausible area, corner ordering, two opposite line families, and a canonical 2:1 playing-surface aspect after rectification.
5. Produce a confidence score and preserve a manual four-corner fallback. Do not run the second pass when the coarse geometry is clearly invalid.

### Pass B — rail-strip sight candidates

For each side, extract a high-resolution band spanning the cushion edge through the rail cap. Rectify each band to a horizontal strip if possible. Generate candidates using a union of:

- the current YOLO sight predictions, including low-confidence centres;
- high-lightness, low-chroma connected components in Lab/HSV;
- a local white-top-hat or difference-of-Gaussians response for small bright features under uneven illumination;
- compact round/diamond contour evidence, with size and aspect limits expressed in normalized rail coordinates.

Then score or reject candidates with:

- distance to the predicted sight row;
- local contrast against the rail immediately around the candidate;
- normalized size, solidity, convexity, and round-or-diamond shape;
- a per-rail one-dimensional lattice fit with the expected 3/6 positions;
- cross-rail consistency in spacing and physical size;
- allowance for missing sights, partial occlusion, and one name plate.

This should be a **hybrid candidate union**, not a classical-vision replacement for YOLO. The Pix2Pockets Hough-circle ablation is evidence that a purely white/circle detector is too brittle in the wild.

### Final refinement and acceptance

1. Back-project the accepted strip candidates to source-image coordinates.
2. Establish their canonical indices with a robust one-dimensional lattice fit per rail.
3. Fit the final homography using all accepted sight correspondences plus consistent inferred corners, using RANSAC or another robust estimator.
4. Recompute inliers and refine with a symmetric reprojection objective.
5. Reject or fall back to manual corners when counts, lattice residual, reprojection error, corner plausibility, or cross-view consistency fail.

This is slightly different from the current `fit_rails` contract in [`src/billiards/geometry.py`](../src/billiards/geometry.py): the present code asks for at least 12 unordered sight points and then discovers four lines. A coarse table stage would supply probable rail membership and a narrow search band before final consensus, while the existing fitter can remain the independent validation/fallback path.

## Smallest useful experiment

Do not retrain another full-frame detector first. Run a validation-only ablation with the current 960 checkpoint and keep the held-out test set sealed:

| Variant | Candidate source | Geometry prior |
|---|---|---|
| A — current baseline | Full-frame YOLO | Existing four-line fitter |
| B — crop only | YOLO on four high-resolution coarse rail crops | Coarse rail membership |
| C — classical constrained | Bright/low-chroma blob candidates in rectified strips | Rail row + size/shape + 3/6 lattice |
| D — recommended hybrid | Union of B and C | Rail row + lattice + final robust refinement |

Measure, per image and per rail:

- sight-centre precision/recall at normalized physical-distance thresholds;
- recall for near versus far rails and by table-area fraction;
- candidate count before and after lattice filtering;
- rails with enough correct support, rather than only aggregate Dot AP;
- correct four-rail recovery rate and source-image corner error;
- final homography reprojection residual and failure/rejection rate;
- matched-view projected ball-position error on the reserved geometry split;
- runtime and memory.

The experiment succeeds only if the hybrid improves geometry success or homography error without turning coarse-table failures into confidently wrong calibrations. A useful early go/no-go result is whether crop-only inference recovers additional far-rail sights on the current worst frames; that isolates the benefit of scale/ROI normalization before building the classical candidate stage.

## Prioritized decision

1. First complete the already-planned 960-pixel Dot-centre and rail-confidence sweeps; they establish the baseline the new cascade must beat.
2. Add a coarse table/cloth quadrilateral **prototype** and evaluate its corner error independently.
3. Try four high-resolution rail crops with the existing detector. This is the cheapest direct test of the user's idea.
4. Add white/low-chroma blob candidates and the rail-wise lattice only if crop-only still misses sights or yields too many false candidates.
5. Use validated sights—not the coarse cloth quadrilateral—as the final calibration refinement whenever enough sights survive.

The core conclusion is therefore: **rail-first is promising and well supported by the ordering used in other billiards vision systems, but it should bootstrap and constrain sight detection, not replace sight-based final calibration.**
