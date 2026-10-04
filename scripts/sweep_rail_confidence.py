"""Sweep YOLO Dot confidence using downstream four-rail correctness.

Unlike a detector-only threshold sweep, this script evaluates whether the Dot
predictions recover the labelled table quadrilateral. YOLO runs once at the lowest
confidence; the same raw prediction set is filtered at every higher threshold.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import math
from pathlib import Path
from statistics import mean, median
from typing import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from analyze_detector_errors import write_csv  # noqa: E402
from fit_table_rails import (  # noqa: E402
    find_images,
    load_labelled_dots,
    load_settings,
    prediction_points,
    rail_config,
    rail_contact_sheet,
    render_result,
    yolo_dot_class_id,
)
from billiards.geometry import fit_rails  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose Dot confidence using downstream rail fitting."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/pix2pockets_v3/valid/images"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/detection/baseline-yolo11n-960/weights/best.pt"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/geometry.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/geometry/rail-confidence-sweep"),
    )
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--dot-class-id", type=int, default=2)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
    )
    return parser.parse_args()


def confidence_slug(confidence: float) -> str:
    return f"conf_{round(confidence * 100):03d}"


def corner_alignment_error(
    predicted: Sequence[tuple[float, float]],
    reference: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """Return the best unordered mean/max corner error in native pixels."""
    if len(predicted) != 4 or len(reference) != 4:
        return None
    best: tuple[float, float] | None = None
    predicted_array = np.asarray(predicted, dtype=float)
    for ordering in permutations(reference):
        distances = np.linalg.norm(
            predicted_array - np.asarray(ordering, dtype=float), axis=1
        )
        candidate = (float(np.mean(distances)), float(np.max(distances)))
        if best is None or candidate < best:
            best = candidate
    return best


def summarize(
    rows_by_threshold: dict[float, list[dict[str, object]]],
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for threshold, rows in rows_by_threshold.items():
        dot_counts = [int(row["dot_count"]) for row in rows]
        corner_errors = [
            float(row["corner_mean_error_px"])
            for row in rows
            if row["corner_mean_error_px"] != ""
        ]
        summary_rows.append(
            {
                "confidence": threshold,
                "images": len(rows),
                "four_rail_count": sum(int(row["rail_count"]) == 4 for row in rows),
                "structurally_valid_count": sum(bool(row["valid"]) for row in rows),
                "agrees_0_5pct_count": sum(
                    bool(row["agrees_0_5pct"]) for row in rows
                ),
                "agrees_1pct_count": sum(
                    bool(row["agrees_1pct"]) for row in rows
                ),
                "valid_and_agrees_1pct_count": sum(
                    bool(row["valid"]) and bool(row["agrees_1pct"]) for row in rows
                ),
                "mean_dot_count": mean(dot_counts) if dot_counts else None,
                "median_dot_count": median(dot_counts) if dot_counts else None,
                "mean_corner_error_px_when_four_rails": (
                    mean(corner_errors) if corner_errors else None
                ),
                "median_corner_error_px_when_four_rails": (
                    median(corner_errors) if corner_errors else None
                ),
                "p95_corner_error_px_when_four_rails": (
                    float(np.percentile(corner_errors, 95)) if corner_errors else None
                ),
            }
        )
    return summary_rows


def plot_summary(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    confidences = [float(row["confidence"]) for row in rows]
    figure, left_axis = plt.subplots(figsize=(9, 5.5))
    left_axis.plot(
        confidences,
        [int(row["four_rail_count"]) for row in rows],
        marker="o",
        label="Four rails produced",
    )
    left_axis.plot(
        confidences,
        [int(row["structurally_valid_count"]) for row in rows],
        marker="o",
        label="Structurally valid",
    )
    left_axis.plot(
        confidences,
        [int(row["valid_and_agrees_1pct_count"]) for row in rows],
        marker="o",
        linewidth=2.5,
        label="Valid + agrees with labels",
    )
    left_axis.set_xlabel("YOLO Dot confidence")
    left_axis.set_ylabel("Images out of 20")
    left_axis.set_ylim(0, max(int(row["images"]) for row in rows) + 1)
    left_axis.grid(alpha=0.25)

    right_axis = left_axis.twinx()
    right_axis.plot(
        confidences,
        [float(row["mean_dot_count"]) for row in rows],
        color="tab:gray",
        linestyle="--",
        marker="x",
        label="Mean Dot candidates",
    )
    right_axis.set_ylabel("Mean Dot candidates per image")

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    figure.suptitle("Rail-fitting confidence sweep (validation only)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def format_optional(value: object, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def markdown_report(
    rows: Sequence[dict[str, object]], image_count: int
) -> str:
    best_count = max(int(row["valid_and_agrees_1pct_count"]) for row in rows)
    best_thresholds = [
        float(row["confidence"])
        for row in rows
        if int(row["valid_and_agrees_1pct_count"]) == best_count
    ]
    lines = [
        "# Rail-fitting confidence sweep",
        "",
        f"- Validation images: `{image_count}`",
        "- Held-out detector test split used: **no**",
        "- YOLO inference passes: `1` at the lowest swept confidence",
        "",
        "Label agreement compares the four predicted rail intersections with the",
        "four intersections fitted from labelled Dots. The 1% rule requires mean",
        "corner error <=1% of the image diagonal and maximum error <=2%.",
        "It is a validation diagnostic, not the final homography threshold.",
        "",
        "| Confidence | Mean dots | Four rails | Structurally valid | Label agreement (0.5%) | Label agreement (1%) | Valid + label agreement | Median corner error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {float(row['confidence']):.2f} | {float(row['mean_dot_count']):.1f} | "
            f"{int(row['four_rail_count'])}/{image_count} | "
            f"{int(row['structurally_valid_count'])}/{image_count} | "
            f"{int(row['agrees_0_5pct_count'])}/{image_count} | "
            f"{int(row['agrees_1pct_count'])}/{image_count} | "
            f"{int(row['valid_and_agrees_1pct_count'])}/{image_count} | "
            f"{format_optional(row['median_corner_error_px_when_four_rails'])} px |"
            for row in rows
        ),
        "",
        f"Maximum valid-and-correct count: `{best_count}/{image_count}` at confidence "
        + ", ".join(f"`{value:.2f}`" for value in best_thresholds)
        + ".",
        "",
        "Inspect each confidence directory's `contact_sheet.jpg` before selecting",
        "an operating point. A lower confidence can increase four-line output while",
        "also making false background alignments more likely.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    thresholds = sorted(set(float(value) for value in args.thresholds))
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Thresholds must be between 0 and 1")
    settings = load_settings(args.config)
    config = rail_config(settings)
    imgsz = args.imgsz if args.imgsz is not None else int(settings["imgsz"])
    image_paths = find_images(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    threshold_directories = {
        threshold: args.output / confidence_slug(threshold)
        for threshold in thresholds
    }
    for directory in threshold_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    dot_class_id = yolo_dot_class_id(model)
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        device=args.device,
        batch=args.batch,
        conf=min(thresholds),
        iou=float(settings["prediction_nms_iou"]),
        save=False,
        verbose=False,
        stream=True,
    )

    per_image_rows: list[dict[str, object]] = []
    rows_by_threshold: dict[float, list[dict[str, object]]] = {
        threshold: [] for threshold in thresholds
    }
    for image_path, prediction in zip(image_paths, predictions, strict=True):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image_size = (image.shape[1], image.shape[0])
        ground_truth_points = load_labelled_dots(
            image_path, image_size, args.labels_dir, args.dot_class_id
        )
        ground_truth_result = fit_rails(
            ground_truth_points, image_size=image_size, config=config
        )
        raw_points = prediction_points(prediction, dot_class_id)
        diagonal = math.hypot(*image_size)

        for threshold in thresholds:
            points = [
                point for point in raw_points if point.confidence >= threshold
            ]
            result = fit_rails(points, image_size=image_size, config=config)
            alignment = corner_alignment_error(
                result.corners, ground_truth_result.corners
            )
            mean_error = alignment[0] if alignment is not None else None
            max_error = alignment[1] if alignment is not None else None
            agrees_0_5pct = bool(
                mean_error is not None
                and max_error is not None
                and mean_error <= 0.005 * diagonal
                and max_error <= 0.010 * diagonal
            )
            agrees_1pct = bool(
                mean_error is not None
                and max_error is not None
                and mean_error <= 0.010 * diagonal
                and max_error <= 0.020 * diagonal
            )
            output_dir = threshold_directories[threshold]
            overlay_name = f"{image_path.stem}_rails.jpg"
            cv2.imwrite(
                str(output_dir / overlay_name),
                render_result(image, result, f"yolo@{threshold:.2f}"),
            )
            assigned_count = sum(
                len(rail.inlier_indices) for rail in result.rails
            )
            row: dict[str, object] = {
                "confidence": threshold,
                "image": image_path.name,
                "dot_count": len(points),
                "rail_count": len(result.rails),
                "rail_supports": "/".join(
                    str(len(rail.inlier_indices)) for rail in result.rails
                ),
                "assigned_dot_count": assigned_count,
                "outlier_count": len(result.outlier_indices),
                "valid": result.valid,
                "corner_mean_error_px": mean_error if mean_error is not None else "",
                "corner_max_error_px": max_error if max_error is not None else "",
                "corner_mean_error_diagonal_fraction": (
                    mean_error / diagonal if mean_error is not None else ""
                ),
                "agrees_0_5pct": agrees_0_5pct,
                "agrees_1pct": agrees_1pct,
                "warnings": ";".join(result.warnings),
                "overlay": overlay_name,
            }
            per_image_rows.append(row)
            rows_by_threshold[threshold].append(row)

    for threshold, rows in rows_by_threshold.items():
        output_dir = threshold_directories[threshold]
        write_csv(output_dir / "per_image.csv", rows)
        rail_contact_sheet(output_dir, rows, output_dir / "contact_sheet.jpg")

    summary_rows = summarize(rows_by_threshold)
    write_csv(args.output / "per_image_all_thresholds.csv", per_image_rows)
    write_csv(args.output / "confidence_summary.csv", summary_rows)
    plot_summary(summary_rows, args.output / "confidence_sweep.png")
    report_path = args.output / "RAIL_CONFIDENCE_SWEEP.md"
    report_path.write_text(
        markdown_report(summary_rows, len(image_paths)), encoding="utf-8"
    )

    best_count = max(
        int(row["valid_and_agrees_1pct_count"]) for row in summary_rows
    )
    best_thresholds = [
        float(row["confidence"])
        for row in summary_rows
        if int(row["valid_and_agrees_1pct_count"]) == best_count
    ]
    print(
        f"images={len(image_paths)} best_valid_and_correct={best_count}/{len(image_paths)} "
        f"thresholds={','.join(f'{value:.2f}' for value in best_thresholds)}"
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
