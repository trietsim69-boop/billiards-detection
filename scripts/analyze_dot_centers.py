"""Evaluate rail-dot detections by centre distance instead of box IoU.

YOLO reports boxes, but the table-geometry stage consumes dot centres. This
diagnostic therefore performs one-to-one matching between predicted and labelled
Dot centres at several pixel tolerances. Distances are expressed at the configured
inference scale (640 px by default), while overlays remain in original-image
coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from scripts.analyze_detector_errors import (
        Detection,
        load_dataset,
        load_ground_truth,
        result_to_predictions,
        safe_ratio,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/analyze_dot_centers.py
    from analyze_detector_errors import (  # type: ignore[no-redef]
        Detection,
        load_dataset,
        load_ground_truth,
        result_to_predictions,
        safe_ratio,
    )


MATCH_COLOR = (60, 190, 60)
GROUND_TRUTH_COLOR = (255, 120, 40)
MISSED_COLOR = (40, 40, 230)
FALSE_POSITIVE_COLOR = (0, 165, 255)


@dataclass(frozen=True)
class CenterMatch:
    ground_truth_index: int
    prediction_index: int
    distance_model_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Dot detection quality by centre distance."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/detection/baseline-yolo11n-640/weights/best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/pix2pockets_v3/data.yaml"),
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detection/baseline-yolo11n-640-dot-centers"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--prediction-iou", type=float, default=0.70)
    parser.add_argument("--min-prediction-conf", type=float, default=0.01)
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50),
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=(4.0, 8.0, 12.0, 16.0),
        help="Centre-distance tolerances in pixels at --imgsz scale.",
    )
    parser.add_argument(
        "--display-confidence",
        type=float,
        default=0.50,
        help="Confidence used for the per-image gallery; not an acceptance rule.",
    )
    parser.add_argument(
        "--display-tolerance",
        type=float,
        default=8.0,
        help="Model-scale pixel tolerance used for the gallery.",
    )
    return parser.parse_args()


def center(detection: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = detection.box_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def inference_gain(width: int, height: int, image_size: int) -> float:
    """Return the isotropic letterbox resize gain into model coordinates."""
    if width <= 0 or height <= 0 or image_size <= 0:
        raise ValueError("Image dimensions and image_size must be positive")
    return image_size / max(width, height)


def center_distance_model_px(
    left: Detection, right: Detection, gain: float
) -> float:
    left_x, left_y = center(left)
    right_x, right_y = center(right)
    return math.hypot(left_x - right_x, left_y - right_y) * gain


def match_centers(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    tolerance_model_px: float,
    gain: float,
) -> tuple[CenterMatch, ...]:
    """Find a deterministic maximum-cardinality one-to-one centre matching.

    Candidate edges are ordered by distance. The augmenting-path matcher can
    reassign an earlier pair, avoiding the false misses that a purely greedy
    nearest-neighbour matcher can create when two predictions compete for a dot.
    """
    if tolerance_model_px < 0:
        raise ValueError("tolerance_model_px must be non-negative")

    distances: dict[tuple[int, int], float] = {}
    adjacency: dict[int, list[int]] = {}
    for gt_index, gt_item in enumerate(ground_truth):
        candidates: list[tuple[float, int]] = []
        for prediction_index, prediction in enumerate(predictions):
            distance = center_distance_model_px(gt_item, prediction, gain)
            if distance <= tolerance_model_px:
                distances[(gt_index, prediction_index)] = distance
                candidates.append((distance, prediction_index))
        adjacency[gt_index] = [
            prediction_index
            for _, prediction_index in sorted(candidates, key=lambda item: (item[0], item[1]))
        ]

    prediction_to_gt: dict[int, int] = {}

    def augment(gt_index: int, visited_predictions: set[int]) -> bool:
        for prediction_index in adjacency[gt_index]:
            if prediction_index in visited_predictions:
                continue
            visited_predictions.add(prediction_index)
            previous_gt = prediction_to_gt.get(prediction_index)
            if previous_gt is None or augment(previous_gt, visited_predictions):
                prediction_to_gt[prediction_index] = gt_index
                return True
        return False

    gt_order = sorted(
        range(len(ground_truth)), key=lambda index: (len(adjacency[index]), index)
    )
    for gt_index in gt_order:
        augment(gt_index, set())

    matches = [
        CenterMatch(
            ground_truth_index=gt_index,
            prediction_index=prediction_index,
            distance_model_px=distances[(gt_index, prediction_index)],
        )
        for prediction_index, gt_index in prediction_to_gt.items()
    ]
    return tuple(
        sorted(matches, key=lambda item: (item.ground_truth_index, item.prediction_index))
    )


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def optional_stat(values: Sequence[float], operation: str) -> float | None:
    if not values:
        return None
    if operation == "mean":
        return mean(values)
    if operation == "median":
        return median(values)
    raise ValueError(f"Unknown operation: {operation}")


def evaluate_setting(
    ground_truth_by_image: Sequence[Sequence[Detection]],
    predictions_by_image: Sequence[Sequence[Detection]],
    gains: Sequence[float],
    confidence: float,
    tolerance_model_px: float,
) -> dict[str, object]:
    total_gt = 0
    total_predictions = 0
    total_matches = 0
    fully_matched_images = 0
    distances: list[float] = []

    for ground_truth, raw_predictions, gain in zip(
        ground_truth_by_image, predictions_by_image, gains
    ):
        predictions = [
            prediction
            for prediction in raw_predictions
            if prediction.confidence >= confidence
        ]
        matches = match_centers(
            ground_truth, predictions, tolerance_model_px, gain
        )
        total_gt += len(ground_truth)
        total_predictions += len(predictions)
        total_matches += len(matches)
        distances.extend(match.distance_model_px for match in matches)
        if len(matches) == len(ground_truth):
            fully_matched_images += 1

    false_positives = total_predictions - total_matches
    false_negatives = total_gt - total_matches
    precision = safe_ratio(total_matches, total_matches + false_positives)
    recall = safe_ratio(total_matches, total_matches + false_negatives)
    return {
        "confidence": confidence,
        "tolerance_model_px": tolerance_model_px,
        "tp": total_matches,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
        "mean_error_model_px": optional_stat(distances, "mean"),
        "median_error_model_px": optional_stat(distances, "median"),
        "p95_error_model_px": percentile(distances, 95),
        "fully_matched_images": fully_matched_images,
        "image_count": len(ground_truth_by_image),
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_marker(
    image: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    marker: int,
) -> None:
    position = (int(round(point[0])), int(round(point[1])))
    thickness = max(2, round(min(image.shape[:2]) / 500))
    cv2.drawMarker(
        image,
        position,
        color,
        markerType=marker,
        markerSize=max(12, thickness * 5),
        thickness=thickness,
    )


def annotate_centers(
    image: np.ndarray,
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    matches: Sequence[CenterMatch],
    confidence: float,
    tolerance_model_px: float,
) -> np.ndarray:
    annotated = image.copy()
    matched_gt = {match.ground_truth_index for match in matches}
    matched_predictions = {match.prediction_index for match in matches}

    for match in matches:
        gt_center = center(ground_truth[match.ground_truth_index])
        prediction_center = center(predictions[match.prediction_index])
        cv2.line(
            annotated,
            tuple(int(round(value)) for value in gt_center),
            tuple(int(round(value)) for value in prediction_center),
            MATCH_COLOR,
            max(1, round(min(image.shape[:2]) / 700)),
            cv2.LINE_AA,
        )
        draw_marker(annotated, gt_center, GROUND_TRUTH_COLOR, cv2.MARKER_TILTED_CROSS)
        draw_marker(annotated, prediction_center, MATCH_COLOR, cv2.MARKER_CROSS)

    for gt_index, detection in enumerate(ground_truth):
        if gt_index not in matched_gt:
            draw_marker(annotated, center(detection), MISSED_COLOR, cv2.MARKER_DIAMOND)

    for prediction_index, detection in enumerate(predictions):
        if prediction_index not in matched_predictions:
            draw_marker(
                annotated,
                center(detection),
                FALSE_POSITIVE_COLOR,
                cv2.MARKER_TRIANGLE_UP,
            )

    summary = (
        f"Dot centres | conf={confidence:.2f} tolerance={tolerance_model_px:g}px@640 "
        f"| match={len(matches)} miss={len(ground_truth) - len(matches)} "
        f"fp={len(predictions) - len(matches)}"
    )
    legend = "blue=GT green=matched red=miss orange=FP"
    scale = max(0.55, min(image.shape[:2]) / 1050)
    thickness = max(1, round(min(image.shape[:2]) / 550))
    text_height = cv2.getTextSize(
        summary, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )[0][1]
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (annotated.shape[1] - 1, text_height * 3 + 18),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.80, annotated, 0.20, 0, annotated)
    cv2.putText(
        annotated,
        summary,
        (8, text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        legend,
        (8, text_height * 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def create_contact_sheet(
    gallery_dir: Path,
    per_image_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    worst_rows = sorted(
        per_image_rows,
        key=lambda row: (int(row["missed"]), int(row["false_positive"])),
        reverse=True,
    )[:12]
    tile_width, tile_height = 480, 300
    tiles: list[np.ndarray] = []
    for row in worst_rows:
        source = cv2.imread(str(gallery_dir / str(row["gallery_file"])))
        if source is None:
            continue
        scale = min(tile_width / source.shape[1], tile_height / source.shape[0])
        resized = cv2.resize(
            source,
            (
                max(1, round(source.shape[1] * scale)),
                max(1, round(source.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.full((tile_height, tile_width, 3), 28, dtype=np.uint8)
        y_offset = (tile_height - resized.shape[0]) // 2
        x_offset = (tile_width - resized.shape[1]) // 2
        tile[
            y_offset : y_offset + resized.shape[0],
            x_offset : x_offset + resized.shape[1],
        ] = resized
        tiles.append(tile)

    if not tiles:
        return
    columns = 3
    blank = np.full_like(tiles[0], 28)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [
        np.hstack(tiles[index : index + columns])
        for index in range(0, len(tiles), columns)
    ]
    cv2.imwrite(str(output_path), np.vstack(rows))


def markdown_report(
    args: argparse.Namespace,
    image_count: int,
    ground_truth_count: int,
    sweep_rows: Sequence[dict[str, object]],
    fixed_rows: Sequence[dict[str, object]],
) -> str:
    def row_text(row: dict[str, object]) -> str:
        return (
            f"| {float(row['tolerance_model_px']):g} | {int(row['tp'])} | "
            f"{int(row['fp'])} | {int(row['fn'])} | "
            f"{float(row['precision']):.3f} | {float(row['recall']):.3f} | "
            f"{float(row['f1']):.3f} | {int(row['fully_matched_images'])}/{image_count} |"
        )

    best_rows = []
    for tolerance in sorted({float(row["tolerance_model_px"]) for row in sweep_rows}):
        candidates = [
            row
            for row in sweep_rows
            if float(row["tolerance_model_px"]) == tolerance
        ]
        best_rows.append(
            max(
                candidates,
                key=lambda row: (
                    float(row["f1"]),
                    float(row["recall"]),
                    -float(row["confidence"]),
                ),
            )
        )

    lines = [
        "# Dot-centre validation diagnostic",
        "",
        f"- Images: `{image_count}`",
        f"- Ground-truth dots: `{ground_truth_count}`",
        f"- Model image size: `{args.imgsz}`",
        "- Test split used: **no**",
        "",
        "A tolerance of 8 px means 8 pixels after the image is letterboxed to the",
        f"model's `{args.imgsz}`-pixel scale. It is a diagnostic tolerance, not yet",
        "a final homography acceptance requirement.",
        "",
        f"## Fixed confidence `{args.display_confidence:.2f}`",
        "",
        "| Tolerance (model px) | TP | FP | FN | Precision | Recall | F1 | Complete images |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(row_text(row) for row in fixed_rows),
        "",
        "## Best F1 confidence at each tolerance",
        "",
        "| Tolerance (model px) | Confidence | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|",
        *(
            f"| {float(row['tolerance_model_px']):g} | {float(row['confidence']):.2f} | "
            f"{float(row['precision']):.3f} | {float(row['recall']):.3f} | "
            f"{float(row['f1']):.3f} |"
            for row in best_rows
        ),
        "",
        "## How to read the gallery",
        "",
        "- Blue tilted cross: labelled Dot centre.",
        "- Green cross and line: matched prediction and its centre error.",
        "- Red diamond: labelled Dot with no prediction inside the tolerance.",
        "- Orange triangle: unmatched Dot prediction.",
        "",
        "The gallery uses the configured display confidence and tolerance. See",
        "`distance_sweep.csv` before choosing a final operating point.",
    ]
    return "\n".join(lines) + "\n"


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.confidence_thresholds):
        raise ValueError("Confidence thresholds must be between 0 and 1")
    if any(value < 0.0 for value in args.tolerances):
        raise ValueError("Tolerances must be non-negative")
    if not 0.0 <= args.display_confidence <= 1.0:
        raise ValueError("--display-confidence must be between 0 and 1")
    if args.display_tolerance < 0:
        raise ValueError("--display-tolerance must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    gallery_dir = args.output / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    image_paths, class_names, image_dir = load_dataset(args.data, args.split)
    if "Dot" not in class_names:
        raise ValueError("Dataset has no Dot class")
    dot_class_id = class_names.index("Dot")

    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf=args.min_prediction_conf,
        iou=args.prediction_iou,
        save=False,
        verbose=False,
    )
    if len(results) != len(image_paths):
        raise RuntimeError(
            f"Expected {len(image_paths)} prediction results, got {len(results)}"
        )

    ground_truth_by_image: list[list[Detection]] = []
    predictions_by_image: list[list[Detection]] = []
    gains: list[float] = []
    images: list[np.ndarray] = []
    for image_path, result in zip(image_paths, results):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        all_ground_truth = load_ground_truth(image_path, width, height)
        all_predictions = result_to_predictions(result)
        ground_truth_by_image.append(
            [item for item in all_ground_truth if item.class_id == dot_class_id]
        )
        predictions_by_image.append(
            [item for item in all_predictions if item.class_id == dot_class_id]
        )
        gains.append(inference_gain(width, height, args.imgsz))
        images.append(image)

    confidences = sorted(set(args.confidence_thresholds))
    tolerances = sorted(set(args.tolerances))
    sweep_rows = [
        evaluate_setting(
            ground_truth_by_image,
            predictions_by_image,
            gains,
            confidence,
            tolerance,
        )
        for confidence in confidences
        for tolerance in tolerances
    ]
    fixed_rows = [
        evaluate_setting(
            ground_truth_by_image,
            predictions_by_image,
            gains,
            args.display_confidence,
            tolerance,
        )
        for tolerance in tolerances
    ]

    per_image_rows: list[dict[str, object]] = []
    matched_pair_rows: list[dict[str, object]] = []
    for image_path, image, ground_truth, raw_predictions, gain in zip(
        image_paths,
        images,
        ground_truth_by_image,
        predictions_by_image,
        gains,
    ):
        predictions = [
            prediction
            for prediction in raw_predictions
            if prediction.confidence >= args.display_confidence
        ]
        matches = match_centers(
            ground_truth, predictions, args.display_tolerance, gain
        )
        distances = [match.distance_model_px for match in matches]
        gallery_name = f"{image_path.stem}_dot_centers.jpg"
        annotated = annotate_centers(
            image,
            ground_truth,
            predictions,
            matches,
            args.display_confidence,
            args.display_tolerance,
        )
        cv2.imwrite(str(gallery_dir / gallery_name), annotated)

        per_image_rows.append(
            {
                "image": image_path.name,
                "gallery_file": gallery_name,
                "ground_truth_dots": len(ground_truth),
                "predicted_dots": len(predictions),
                "matched": len(matches),
                "missed": len(ground_truth) - len(matches),
                "false_positive": len(predictions) - len(matches),
                "recall": safe_ratio(len(matches), len(ground_truth)),
                "mean_error_model_px": optional_stat(distances, "mean"),
                "p95_error_model_px": percentile(distances, 95),
                "complete": len(matches) == len(ground_truth),
            }
        )
        for match in matches:
            gt_x, gt_y = center(ground_truth[match.ground_truth_index])
            prediction = predictions[match.prediction_index]
            prediction_x, prediction_y = center(prediction)
            matched_pair_rows.append(
                {
                    "image": image_path.name,
                    "gt_x_original_px": gt_x,
                    "gt_y_original_px": gt_y,
                    "prediction_x_original_px": prediction_x,
                    "prediction_y_original_px": prediction_y,
                    "distance_original_px": match.distance_model_px / gain,
                    "distance_model_px": match.distance_model_px,
                    "prediction_confidence": prediction.confidence,
                }
            )

    write_csv(args.output / "distance_sweep.csv", sweep_rows)
    write_csv(args.output / "fixed_confidence.csv", fixed_rows)
    write_csv(args.output / "per_image.csv", per_image_rows)
    write_csv(args.output / "matched_pairs.csv", matched_pair_rows)
    create_contact_sheet(
        gallery_dir, per_image_rows, args.output / "worst_cases.jpg"
    )

    ground_truth_count = sum(len(items) for items in ground_truth_by_image)
    summary = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "image_directory": str(image_dir),
        "image_count": len(image_paths),
        "ground_truth_dot_count": ground_truth_count,
        "image_size": args.imgsz,
        "prediction_nms_iou": args.prediction_iou,
        "min_prediction_confidence": args.min_prediction_conf,
        "confidence_thresholds": confidences,
        "tolerances_model_px": tolerances,
        "display_confidence": args.display_confidence,
        "display_tolerance_model_px": args.display_tolerance,
        "fixed_confidence_metrics": fixed_rows,
        "test_split_used": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report_path = args.output / "DOT_CENTER_REPORT.md"
    report_path.write_text(
        markdown_report(
            args,
            len(image_paths),
            ground_truth_count,
            sweep_rows,
            fixed_rows,
        ),
        encoding="utf-8",
    )

    display_row = evaluate_setting(
        ground_truth_by_image,
        predictions_by_image,
        gains,
        args.display_confidence,
        args.display_tolerance,
    )
    print(
        f"images={len(image_paths)} dots={ground_truth_count} "
        f"confidence={args.display_confidence:.2f} "
        f"tolerance={args.display_tolerance:g}px@{args.imgsz}"
    )
    print(
        f"precision={float(display_row['precision']):.4f} "
        f"recall={float(display_row['recall']):.4f} "
        f"f1={float(display_row['f1']):.4f} "
        f"complete_images={display_row['fully_matched_images']}/{len(image_paths)}"
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
