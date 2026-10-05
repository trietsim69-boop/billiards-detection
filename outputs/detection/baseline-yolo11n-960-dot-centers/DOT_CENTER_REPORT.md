# Dot-centre validation diagnostic

- Images: `20`
- Ground-truth dots: `359`
- Model image size: `960`
- Test split used: **no**

A tolerance of 8 px means 8 pixels after the image is letterboxed to the
model's `960`-pixel scale. It is a diagnostic tolerance, not yet
a final homography acceptance requirement.

## Fixed confidence `0.50`

| Tolerance (model px) | TP | FP | FN | Precision | Recall | F1 | Complete images |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 314 | 4 | 45 | 0.987 | 0.875 | 0.928 | 11/20 |
| 8 | 314 | 4 | 45 | 0.987 | 0.875 | 0.928 | 11/20 |
| 12 | 315 | 3 | 44 | 0.991 | 0.877 | 0.931 | 11/20 |
| 16 | 315 | 3 | 44 | 0.991 | 0.877 | 0.931 | 11/20 |

## Best F1 confidence at each tolerance

| Tolerance (model px) | Confidence | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 4 | 0.20 | 0.959 | 0.908 | 0.933 |
| 8 | 0.20 | 0.959 | 0.908 | 0.933 |
| 12 | 0.20 | 0.962 | 0.911 | 0.936 |
| 16 | 0.20 | 0.962 | 0.911 | 0.936 |

## How to read the gallery

- Blue tilted cross: labelled Dot centre.
- Green cross and line: matched prediction and its centre error.
- Red diamond: labelled Dot with no prediction inside the tolerance.
- Orange triangle: unmatched Dot prediction.

The gallery uses the configured display confidence and tolerance. See
`distance_sweep.csv` before choosing a final operating point.
