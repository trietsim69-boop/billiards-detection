# Rail-fitting confidence sweep

- Validation images: `20`
- Held-out detector test split used: **no**
- YOLO inference passes: `1` at the lowest swept confidence

Label agreement compares the four predicted rail intersections with the
four intersections fitted from labelled Dots. The 1% rule requires mean
corner error <=1% of the image diagonal and maximum error <=2%.
It is a validation diagnostic, not the final homography threshold.

| Confidence | Mean dots | Four rails | Structurally valid | Label agreement (0.5%) | Label agreement (1%) | Valid + label agreement | Median corner error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 32.5 | 20/20 | 19/20 | 19/20 | 19/20 | 19/20 | 2.03 px |
| 0.10 | 28.6 | 20/20 | 19/20 | 19/20 | 19/20 | 19/20 | 1.87 px |
| 0.15 | 25.8 | 20/20 | 19/20 | 19/20 | 19/20 | 19/20 | 1.89 px |
| 0.20 | 23.4 | 20/20 | 19/20 | 19/20 | 19/20 | 19/20 | 1.74 px |
| 0.25 | 22.0 | 19/20 | 18/20 | 18/20 | 18/20 | 18/20 | 1.57 px |
| 0.30 | 19.9 | 18/20 | 18/20 | 18/20 | 18/20 | 18/20 | 1.54 px |
| 0.40 | 16.7 | 14/20 | 14/20 | 14/20 | 14/20 | 14/20 | 1.34 px |
| 0.50 | 13.3 | 9/20 | 9/20 | 9/20 | 9/20 | 9/20 | 0.67 px |

Maximum valid-and-correct count: `19/20` at confidence `0.05`, `0.10`, `0.15`, `0.20`.

Inspect each confidence directory's `contact_sheet.jpg` before selecting
an operating point. A lower confidence can increase four-line output while
also making false background alignments more likely.
