"""Fail fast unless a detector memorizes the diagnostic tiny dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/datasets/pix2pockets_tiny_overfit.yaml"),
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--min-map50", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = YOLO(str(args.weights)).val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=1,
        device=0,
        workers=0,
        plots=False,
        verbose=False,
    )
    precision, recall, map50, map50_95 = map(float, metrics.box.mean_results())
    print(
        f"precision={precision:.4f} recall={recall:.4f} "
        f"map50={map50:.4f} map50_95={map50_95:.4f}"
    )
    if map50 < args.min_map50 or recall < args.min_recall:
        print(
            "TINY_OVERFIT_CHECK=FAIL "
            f"(required map50>={args.min_map50:.2f}, recall>={args.min_recall:.2f})"
        )
        return 1
    print("TINY_OVERFIT_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
