# Pix2Pockets v3 split

Built from the Roboflow export [8-Ball Pool v3](https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o/dataset/3) (CC BY 4.0). The raw export was removed from this repo; it remains in git history before the final-ship commit.

- The 195 main images are split into 155 train, 20 validation, and 20 test images.
- Conservative dHash grouping keeps near-duplicate images in one split.
- The 52 `a`, `f`, and `t` images are matched views of 25 situations, kept apart in `geometry_eval/`.
- Valid segmentation polygon rows are converted to tight YOLO detection boxes.
- `split_manifest.csv` records provenance, hashes, grouping, and repairs; `audit_report.json` records the label audit.

Do not edit this directory by hand.
