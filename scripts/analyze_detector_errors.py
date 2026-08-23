"""Analyze YOLO detection errors on a labelled split and build a visual gallery.

The script intentionally defaults to the validation split. It loads predictions at
a low confidence, sweeps several confidence thresholds, selects the threshold that
maximizes Dot F1, and then classifies detections as correct, missed, false positive,
or wrong class using class-aware IoU matching.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {
    "correct": (60, 190, 60),
    "missed": (40, 40, 230),
    "false_positive": (0, 165, 255),
    "wrong_class": (190, 50, 190),
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    box_xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass(frozen=True)
class MatchResult:
    correct: tuple[tuple[int, int], ...]
    wrong_class: tuple[tuple[int, int], ...]
    false_positive: tuple[int, ...]
    missed: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validation metrics and a YOLO error gallery."
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
        default=Path("outputs/detection/baseline-yolo11n-640-error-analysis"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--prediction-iou", type=float, default=0.70)
    parser.add_argument("--min-prediction-conf", type=float, default=0.01)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70),
    )
    return parser.parse_args()


def load_dataset(
    data_path: Path, split: str
) -> tuple[list[Path], list[str], Path]:
    data_path = data_path.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    split_value = config[split]
    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = data_path.parent / image_dir
    image_dir = image_dir.resolve()

    names_value = config["names"]
    if isinstance(names_value, dict):
        class_names = [str(names_value[index]) for index in range(len(names_value))]
    else:
        class_names = [str(name) for name in names_value]

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return image_paths, class_names, image_dir


def label_path_for(image_path: Path) -> Path:
    if image_path.parent.name != "images":
        raise ValueError(f"Expected an images directory, got {image_path.parent}")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def load_ground_truth(image_path: Path, width: int, height: int) -> list[Detection]:
    label_path = label_path_for(image_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

    detections: list[Detection] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {label_path}:{line_number}")
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        x1 = (center_x - box_width / 2.0) * width
        y1 = (center_y - box_height / 2.0) * height
        x2 = (center_x + box_width / 2.0) * width
        y2 = (center_y + box_height / 2.0) * height
        detections.append(Detection(class_id, (x1, y1, x2, y2)))
    return detections


def result_to_predictions(result: object) -> list[Detection]:
    boxes = getattr(result, "boxes")
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    return [
        Detection(int(class_id), tuple(map(float, box)), float(confidence))
        for box, class_id, confidence in zip(xyxy, classes, confidences)
    ]


def box_iou(left: Detection, right: Detection) -> float:
    lx1, ly1, lx2, ly2 = left.box_xyxy
    rx1, ry1, rx2, ry2 = right.box_xyxy
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_pairs(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    gt_indices: Iterable[int],
    prediction_indices: Iterable[int],
    iou_threshold: float,
    require_same_class: bool,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, float, int, int]] = []
    for gt_index in gt_indices:
        for prediction_index in prediction_indices:
            same_class = (
                ground_truth[gt_index].class_id
                == predictions[prediction_index].class_id
            )
            if same_class != require_same_class:
                continue
            overlap = box_iou(ground_truth[gt_index], predictions[prediction_index])
            if overlap >= iou_threshold:
                candidates.append(
                    (
                        overlap,
                        predictions[prediction_index].confidence,
                        gt_index,
                        prediction_index,
                    )
                )

    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, _, gt_index, prediction_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append((gt_index, prediction_index))
    return matches


def match_detections(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    iou_threshold: float,
) -> MatchResult:
    all_gt = set(range(len(ground_truth)))
    all_predictions = set(range(len(predictions)))
    correct = greedy_pairs(
        ground_truth,
        predictions,
        all_gt,
        all_predictions,
        iou_threshold,
        require_same_class=True,
    )
    matched_gt = {gt_index for gt_index, _ in correct}
    matched_predictions = {prediction_index for _, prediction_index in correct}

    remaining_gt = all_gt - matched_gt
    remaining_predictions = all_predictions - matched_predictions
    wrong_class = greedy_pairs(
        ground_truth,
        predictions,
        remaining_gt,
        remaining_predictions,
        iou_threshold,
        require_same_class=False,
    )
    matched_gt.update(gt_index for gt_index, _ in wrong_class)
    matched_predictions.update(prediction_index for _, prediction_index in wrong_class)
    return MatchResult(
        correct=tuple(correct),
        wrong_class=tuple(wrong_class),
        false_positive=tuple(sorted(all_predictions - matched_predictions)),
        missed=tuple(sorted(all_gt - matched_gt)),
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics_for_threshold(
    ground_truth_by_image: Sequence[Sequence[Detection]],
    predictions_by_image: Sequence[Sequence[Detection]],
    threshold: float,
    iou_threshold: float,
    class_names: Sequence[str],
) -> list[dict[str, object]]:
    counts = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in range(len(class_names))
    }
    for ground_truth, raw_predictions in zip(
        ground_truth_by_image, predictions_by_image
    ):
        predictions = [
            prediction
            for prediction in raw_predictions
            if prediction.confidence >= threshold
        ]
        matches = match_detections(ground_truth, predictions, iou_threshold)
        for gt_index, _ in matches.correct:
            counts[ground_truth[gt_index].class_id]["tp"] += 1
        for prediction_index in matches.false_positive:
            counts[predictions[prediction_index].class_id]["fp"] += 1
        for gt_index, prediction_index in matches.wrong_class:
            counts[ground_truth[gt_index].class_id]["fn"] += 1
            counts[predictions[prediction_index].class_id]["fp"] += 1
        for gt_index in matches.missed:
            counts[ground_truth[gt_index].class_id]["fn"] += 1

    rows: list[dict[str, object]] = []
    for class_id, class_name in enumerate(class_names):
        tp = counts[class_id]["tp"]
        fp = counts[class_id]["fp"]
        fn = counts[class_id]["fn"]
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        rows.append(
            {
                "threshold": threshold,
                "class_id": class_id,
                "class_name": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    total_tp = sum(value["tp"] for value in counts.values())
    total_fp = sum(value["fp"] for value in counts.values())
    total_fn = sum(value["fn"] for value in counts.values())
    total_precision = safe_ratio(total_tp, total_tp + total_fp)
    total_recall = safe_ratio(total_tp, total_tp + total_fn)
    rows.append(
        {
            "threshold": threshold,
            "class_id": -1,
            "class_name": "all",
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_precision,
            "recall": total_recall,
            "f1": safe_ratio(
                2 * total_precision * total_recall,
                total_precision + total_recall,
            ),
        }
    )
    return rows


def select_dot_threshold(
    sweep_rows: Sequence[dict[str, object]], class_names: Sequence[str]
) -> float:
    if "Dot" not in class_names:
        raise ValueError("Dataset has no Dot class")
    dot_rows = [row for row in sweep_rows if row["class_name"] == "Dot"]
    best = max(
        dot_rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            -abs(float(row["threshold"]) - 0.25),
        ),
    )
    return float(best["threshold"])


def size_group(detection: Detection) -> str:
    x1, y1, x2, y2 = detection.box_xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area < 32**2:
        return "small_lt_32px"
    if area < 96**2:
        return "medium_32_to_96px"
    return "large_ge_96px"


def size_recall_rows(
    ground_truth_by_image: Sequence[Sequence[Detection]],
    predictions_by_image: Sequence[Sequence[Detection]],
    threshold: float,
    iou_threshold: float,
    class_names: Sequence[str],
) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for ground_truth, raw_predictions in zip(
        ground_truth_by_image, predictions_by_image
    ):
        predictions = [p for p in raw_predictions if p.confidence >= threshold]
        matches = match_detections(ground_truth, predictions, iou_threshold)
        matched_gt = {gt_index for gt_index, _ in matches.correct}
        for gt_index, detection in enumerate(ground_truth):
            key = (class_names[detection.class_id], size_group(detection))
            counts.setdefault(key, {"gt": 0, "matched": 0})
            counts[key]["gt"] += 1
            if gt_index in matched_gt:
                counts[key]["matched"] += 1
    return [
        {
            "class_name": class_name,
            "size_group": group,
            "ground_truth": values["gt"],
            "matched": values["matched"],
            "recall": safe_ratio(values["matched"], values["gt"]),
        }
        for (class_name, group), values in sorted(counts.items())
    ]


def draw_box(
    image: np.ndarray,
    detection: Detection,
    text: str,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (
        int(round(value)) for value in detection.box_xyxy
    )
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    thickness = max(2, round(min(width, height) / 450))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.drawMarker(
        image,
        center,
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=max(8, thickness * 4),
        thickness=thickness,
    )
    font_scale = max(0.42, min(width, height) / 1300)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_y = max(text_height + baseline + 2, y1)
    cv2.rectangle(
        image,
        (x1, text_y - text_height - baseline - 4),
        (min(width - 1, x1 + text_width + 4), text_y + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 2, text_y - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate_image(
    image: np.ndarray,
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    matches: MatchResult,
    class_names: Sequence[str],
    threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    annotated = image.copy()
    for _, prediction_index in matches.correct:
        prediction = predictions[prediction_index]
        draw_box(
            annotated,
            prediction,
            f"TP {class_names[prediction.class_id]} {prediction.confidence:.2f}",
            COLORS["correct"],
        )
    for gt_index, prediction_index in matches.wrong_class:
        expected = class_names[ground_truth[gt_index].class_id]
        prediction = predictions[prediction_index]
        actual = class_names[prediction.class_id]
        draw_box(
            annotated,
            prediction,
            f"WRONG {actual}->{expected} {prediction.confidence:.2f}",
            COLORS["wrong_class"],
        )
    for prediction_index in matches.false_positive:
        prediction = predictions[prediction_index]
        draw_box(
            annotated,
            prediction,
            f"FP {class_names[prediction.class_id]} {prediction.confidence:.2f}",
            COLORS["false_positive"],
        )
    for gt_index in matches.missed:
        missed = ground_truth[gt_index]
        draw_box(
            annotated,
            missed,
            f"MISS {class_names[missed.class_id]}",
            COLORS["missed"],
        )

    correct_count = len(matches.correct)
    error_count = (
        len(matches.wrong_class)
        + len(matches.false_positive)
        + len(matches.missed)
    )
    summary = (
        f"conf={threshold:.2f} IoU={iou_threshold:.2f} | correct={correct_count} "
        f"errors={error_count} | green=TP red=miss orange=FP purple=wrong"
    )
    scale = max(0.50, min(image.shape[:2]) / 1100)
    thickness = max(1, round(min(image.shape[:2]) / 550))
    (text_width, text_height), baseline = cv2.getTextSize(
        summary, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (min(annotated.shape[1] - 1, text_width + 12), text_height + baseline + 12),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)
    cv2.putText(
        annotated,
        summary,
        (6, text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_contact_sheet(
    gallery_dir: Path,
    per_image_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    worst_rows = sorted(
        per_image_rows,
        key=lambda row: (
            int(row["missed_dot"]),
            int(row["total_errors"]),
        ),
        reverse=True,
    )[:12]
    tiles: list[np.ndarray] = []
    tile_width, tile_height = 480, 300
    for row in worst_rows:
        source = cv2.imread(str(gallery_dir / str(row["gallery_file"])))
        if source is None:
            continue
        scale = min(tile_width / source.shape[1], tile_height / source.shape[0])
        resized = cv2.resize(
            source,
            (max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale))),
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
    rows = [np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    cv2.imwrite(str(output_path), np.vstack(rows))


def markdown_report(
    args: argparse.Namespace,
    image_count: int,
    object_count: int,
    selected_threshold: float,
    selected_metrics: Sequence[dict[str, object]],
    size_rows: Sequence[dict[str, object]],
    per_image_rows: Sequence[dict[str, object]],
) -> str:
    def metric_row(row: dict[str, object]) -> str:
        return (
            f"| {row['class_name']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{float(row['precision']):.3f} | {float(row['recall']):.3f} | "
            f"{float(row['f1']):.3f} |"
        )

    lines = [
        "# YOLO11n Validation Error Gallery",
        "",
        f"- Images: `{image_count}`",
        f"- Ground-truth objects: `{object_count}`",
        f"- Matching IoU: `{args.iou:.2f}`",
        f"- Selected confidence: `{selected_threshold:.2f}` (highest Dot F1 in the configured sweep)",
        "- Test split used: **no**",
        "",
        "The selected confidence is a diagnostic validation choice, not yet the final inference contract.",
        "",
        "## Selected-threshold metrics",
        "",
        "| Class | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[metric_row(row) for row in selected_metrics],
        "",
        "## Recall by ground-truth box size",
        "",
        "COCO-style pixel-area groups are used: small `<32x32`, medium `32x32–96x96`, and large `>=96x96`.",
        "",
        "| Class | Size group | GT | Matched | Recall |",
        "|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['class_name']} | {row['size_group']} | {row['ground_truth']} | "
        f"{row['matched']} | {float(row['recall']):.3f} |"
        for row in size_rows
    )
    lines.extend(
        [
            "",
            "## Visual legend",
            "",
            "- Green: correct class and IoU match.",
            "- Red: missed ground-truth object.",
            "- Orange: unmatched prediction / false positive.",
            "- Purple: prediction overlaps an object but has the wrong class.",
            "",
            "## Worst cases",
            "",
            "![Worst validation cases](worst_cases.jpg)",
            "",
            "## Per-image gallery",
            "",
            "Images are ordered by missed Dot count, then total error count.",
            "",
            "| Image | GT dots | Predicted dots | Correct dots | Missed dots | False dot predictions | All errors |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        per_image_rows,
        key=lambda value: (
            int(value["missed_dot"]),
            int(value["total_errors"]),
        ),
        reverse=True,
    ):
        lines.append(
            f"| [{row['image']}](gallery/{row['gallery_file']}) | "
            f"{row['gt_dot']} | {row['predicted_dot']} | {row['correct_dot']} | "
            f"{row['missed_dot']} | {row['false_positive_dot']} | "
            f"{row['total_errors']} |"
        )
    lines.extend(
        [
            "",
            "## Machine-readable outputs",
            "",
            "- `threshold_sweep.csv`: class metrics at every confidence threshold.",
            "- `selected_threshold_metrics.csv`: metrics used for this gallery.",
            "- `size_recall.csv`: ground-truth recall grouped by object size.",
            "- `per_image.csv`: image-level errors and dot counts.",
            "- `summary.json`: run configuration and summary metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1]")
    thresholds = sorted(set(args.thresholds))
    if not thresholds or thresholds[0] < args.min_prediction_conf:
        raise ValueError("All sweep thresholds must be >= --min-prediction-conf")

    image_paths, class_names, image_dir = load_dataset(args.data, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    gallery_dir = args.output / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    model_results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.min_prediction_conf,
        iou=args.prediction_iou,
        device=args.device,
        batch=args.batch,
        workers=0,
        save=False,
        verbose=False,
    )
    if len(model_results) != len(image_paths):
        raise RuntimeError("Ultralytics returned a different result count than inputs")

    ground_truth_by_image: list[list[Detection]] = []
    predictions_by_image: list[list[Detection]] = []
    images: list[np.ndarray] = []
    for image_path, result in zip(image_paths, model_results):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV could not load {image_path}")
        height, width = image.shape[:2]
        images.append(image)
        ground_truth_by_image.append(load_ground_truth(image_path, width, height))
        predictions_by_image.append(result_to_predictions(result))

    sweep_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        sweep_rows.extend(
            metrics_for_threshold(
                ground_truth_by_image,
                predictions_by_image,
                threshold,
                args.iou,
                class_names,
            )
        )
    selected_threshold = select_dot_threshold(sweep_rows, class_names)
    selected_metrics = [
        row for row in sweep_rows if float(row["threshold"]) == selected_threshold
    ]
    selected_size_rows = size_recall_rows(
        ground_truth_by_image,
        predictions_by_image,
        selected_threshold,
        args.iou,
        class_names,
    )

    dot_class_id = class_names.index("Dot")
    per_image_rows: list[dict[str, object]] = []
    for image_path, image, ground_truth, raw_predictions in zip(
        image_paths, images, ground_truth_by_image, predictions_by_image
    ):
        predictions = [
            prediction
            for prediction in raw_predictions
            if prediction.confidence >= selected_threshold
        ]
        matches = match_detections(ground_truth, predictions, args.iou)
        gallery_name = f"{image_path.stem}_errors.jpg"
        annotated = annotate_image(
            image,
            ground_truth,
            predictions,
            matches,
            class_names,
            selected_threshold,
            args.iou,
        )
        if not cv2.imwrite(str(gallery_dir / gallery_name), annotated):
            raise RuntimeError(f"Failed to write {gallery_dir / gallery_name}")

        correct_dot = sum(
            ground_truth[gt_index].class_id == dot_class_id
            for gt_index, _ in matches.correct
        )
        missed_dot = sum(
            ground_truth[gt_index].class_id == dot_class_id
            for gt_index in matches.missed
        ) + sum(
            ground_truth[gt_index].class_id == dot_class_id
            for gt_index, _ in matches.wrong_class
        )
        false_positive_dot = sum(
            predictions[prediction_index].class_id == dot_class_id
            for prediction_index in matches.false_positive
        ) + sum(
            predictions[prediction_index].class_id == dot_class_id
            for _, prediction_index in matches.wrong_class
        )
        total_errors = (
            len(matches.missed)
            + len(matches.false_positive)
            + len(matches.wrong_class)
        )
        per_image_rows.append(
            {
                "image": image_path.name,
                "gallery_file": gallery_name,
                "ground_truth_total": len(ground_truth),
                "predicted_total": len(predictions),
                "correct_total": len(matches.correct),
                "wrong_class": len(matches.wrong_class),
                "false_positive": len(matches.false_positive),
                "missed": len(matches.missed),
                "total_errors": total_errors,
                "gt_dot": sum(item.class_id == dot_class_id for item in ground_truth),
                "predicted_dot": sum(
                    item.class_id == dot_class_id for item in predictions
                ),
                "correct_dot": correct_dot,
                "missed_dot": missed_dot,
                "false_positive_dot": false_positive_dot,
            }
        )

    write_csv(args.output / "threshold_sweep.csv", sweep_rows)
    write_csv(args.output / "selected_threshold_metrics.csv", selected_metrics)
    write_csv(args.output / "size_recall.csv", selected_size_rows)
    write_csv(args.output / "per_image.csv", per_image_rows)
    create_contact_sheet(gallery_dir, per_image_rows, args.output / "worst_cases.jpg")

    object_count = sum(len(items) for items in ground_truth_by_image)
    summary = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "image_directory": str(image_dir),
        "image_count": len(image_paths),
        "ground_truth_count": object_count,
        "image_size": args.imgsz,
        "matching_iou": args.iou,
        "prediction_nms_iou": args.prediction_iou,
        "min_prediction_confidence": args.min_prediction_conf,
        "swept_confidence_thresholds": thresholds,
        "selected_confidence_threshold": selected_threshold,
        "threshold_selection_rule": "maximum Dot F1; ties prefer recall then proximity to 0.25",
        "selected_metrics": selected_metrics,
        "test_split_used": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "ERROR_GALLERY.md").write_text(
        markdown_report(
            args,
            len(image_paths),
            object_count,
            selected_threshold,
            selected_metrics,
            selected_size_rows,
            per_image_rows,
        ),
        encoding="utf-8",
    )

    dot_metrics = next(
        row for row in selected_metrics if row["class_name"] == "Dot"
    )
    print(f"images={len(image_paths)} ground_truth={object_count}")
    print(
        f"selected_confidence={selected_threshold:.2f} "
        f"dot_precision={float(dot_metrics['precision']):.4f} "
        f"dot_recall={float(dot_metrics['recall']):.4f} "
        f"dot_f1={float(dot_metrics['f1']):.4f}"
    )
    print(f"report={args.output / 'ERROR_GALLERY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
