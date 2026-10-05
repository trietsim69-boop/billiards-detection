# Dot-centre validation diagnostic

- Images: `20`
- Ground-truth dots: `359`
- Model image size: `1920`
- Test split used: **no**

A tolerance of 8 px means 8 pixels after the image is letterboxed to the
model's `1920`-pixel scale. It is a diagnostic tolerance, not yet
a final homography acceptance requirement.

## Fixed confidence `0.50`

| Tolerance (model px) | TP | FP | FN | Precision | Recall | F1 | Complete images |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 262 | 4 | 97 | 0.985 | 0.730 | 0.838 | 7/20 |

## Best F1 confidence at each tolerance

| Tolerance (model px) | Confidence | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 16 | 0.40 | 0.958 | 0.891 | 0.924 |

## How to read the gallery

- Blue tilted cross: labelled Dot centre.
- Green cross and line: matched prediction and its centre error.
- Red diamond: labelled Dot with no prediction inside the tolerance.
- Orange triangle: unmatched Dot prediction.

The gallery uses the configured display confidence and tolerance. See
`distance_sweep.csv` before choosing a final operating point.
