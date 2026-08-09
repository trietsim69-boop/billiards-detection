# Generated Pix2Pockets v3 split

This directory is generated from the immutable Roboflow export in
`8-Ball Pool.v3i.yolov11`.

- The 195 main images are split into 155 train, 20 validation, and 20 test images.
- Conservative dHash grouping keeps near-duplicate images in one split.
- The 52 `a`, `f`, and `t` images are reserved for geometry evaluation and grouped by their 25 situation identifiers.
- Valid segmentation polygon rows are converted to tight YOLO detection boxes in this processed copy.
- `split_manifest.csv` records provenance, hashes, grouping, and repairs.

Do not edit this directory manually. Regenerate it with `scripts/prepare_dataset.py`.
