"""Fit and visualize four table rails from labelled or YOLO-predicted dots.

Examples:
    # Ground-truth correctness check on the detector validation images
    python scripts/fit_table_rails.py \
        --source data/processed/pix2pockets_v3/valid/images --mode labels

    # A new unlabeled photograph or folder
    python scripts/fit_table_rails.py --source path/to/image.jpg --mode yolo
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from billiards.geometry import (  # noqa: E402
    PointObservation,
    RailFitConfig,
    RailFitResult,
    RailLine,
    fit_rails,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAIL_COLORS = (
    (70, 210, 255),
    (80, 210, 80),
    (255, 140, 60),
    (210, 80, 210),
)
OUTLIER_COLOR = (30, 30, 230)
CORNER_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover four table rails from Dot centres."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("labels", "yolo"),
        default="yolo",
        help="Use YOLO for unlabeled real images; labels is the geometry oracle.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/detection/baseline-yolo11n-640/weights/best.pt"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/geometry.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/geometry/rail-fitting"),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Optional labels directory; otherwise use the images/../labels convention.",
    )
    parser.add_argument("--dot-class-id", type=int, default=2)
    parser.add_argument(
        "--confidence",
        type=float,
        help="Override prediction_confidence from configs/geometry.yaml.",
    )
    parser.add_argument("--imgsz", type=int, help="Override imgsz from config.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def load_settings(path: Path) -> dict[str, object]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return values


def rail_config(settings: dict[str, object]) -> RailFitConfig:
    return RailFitConfig(
        distance_threshold_ratio=float(settings["rail_distance_threshold_ratio"]),
        min_inliers_per_rail=int(settings["min_inliers_per_rail"]),
        min_total_points=int(settings["min_total_points"]),
        min_quad_area_fraction=float(settings["min_quad_area_fraction"]),
        max_quad_area_fraction=float(settings["max_quad_area_fraction"]),
        corner_margin_ratio=float(settings["corner_margin_ratio"]),
        min_adjacent_angle_degrees=float(
            settings["min_adjacent_angle_degrees"]
        ),
    )


def find_images(source: Path) -> list[Path]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")
    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No supported images found in {source}")
    return images


def label_path_for(image_path: Path, labels_dir: Path | None) -> Path:
    if labels_dir is not None:
        return labels_dir.resolve() / f"{image_path.stem}.txt"
    if image_path.parent.name != "images":
        raise ValueError(
            "Label mode expects an images directory or an explicit --labels-dir"
        )
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def load_labelled_dots(
    image_path: Path,
    image_size: tuple[int, int],
    labels_dir: Path | None,
    dot_class_id: int,
) -> list[PointObservation]:
    label_path = label_path_for(image_path, labels_dir)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")
    width, height = image_size
    points: list[PointObservation] = []
    for source_index, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines()
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label row in {label_path}")
        class_id = int(fields[0])
        if class_id != dot_class_id:
            continue
        center_x, center_y = map(float, fields[1:3])
        points.append(
            PointObservation(
                x=center_x * width,
                y=center_y * height,
                confidence=1.0,
                source_index=source_index,
            )
        )
    return points


def yolo_dot_class_id(model: object) -> int:
    names = getattr(model, "names")
    if isinstance(names, dict):
        matches = [index for index, name in names.items() if str(name) == "Dot"]
    else:
        matches = [index for index, name in enumerate(names) if str(name) == "Dot"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Dot class, found {matches}")
    return int(matches[0])


def prediction_points(result: object, dot_class_id: int) -> list[PointObservation]:
    boxes = getattr(result, "boxes")
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    points: list[PointObservation] = []
    for source_index, (box, class_id, confidence) in enumerate(
        zip(xyxy, classes, confidences)
    ):
        if int(class_id) != dot_class_id:
            continue
        x1, y1, x2, y2 = map(float, box)
        points.append(
            PointObservation(
                x=(x1 + x2) / 2.0,
                y=(y1 + y2) / 2.0,
                confidence=float(confidence),
                source_index=source_index,
            )
        )
    return points


def line_image_segment(
    line: RailLine, width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[float, float]] = []
    if abs(line.b) > 1e-12:
        for x in (0.0, float(width - 1)):
            y = -(line.a * x + line.c) / line.b
            if 0.0 <= y <= height - 1:
                candidates.append((x, y))
    if abs(line.a) > 1e-12:
        for y in (0.0, float(height - 1)):
            x = -(line.b * y + line.c) / line.a
            if 0.0 <= x <= width - 1:
                candidates.append((x, y))
    unique: list[tuple[float, float]] = []
    for candidate in candidates:
        if not any(
            np.linalg.norm(np.asarray(candidate) - np.asarray(existing)) < 1e-6
            for existing in unique
        ):
            unique.append(candidate)
    if len(unique) < 2:
        return None
    pair = max(
        (
            (left, right)
            for left_index, left in enumerate(unique)
            for right in unique[left_index + 1 :]
        ),
        key=lambda value: np.linalg.norm(
            np.asarray(value[0]) - np.asarray(value[1])
        ),
    )
    return (
        tuple(int(round(value)) for value in pair[0]),
        tuple(int(round(value)) for value in pair[1]),
    )


def render_result(image: np.ndarray, result: RailFitResult, mode: str) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    thickness = max(2, round(min(width, height) / 500))
    point_radius = max(5, thickness * 2)

    point_to_rail = {
        point_index: rail_index
        for rail_index, rail in enumerate(result.rails)
        for point_index in rail.inlier_indices
    }
    for point_index, point in enumerate(result.points):
        rail_index = point_to_rail.get(point_index)
        color = (
            RAIL_COLORS[rail_index]
            if rail_index is not None
            else OUTLIER_COLOR
        )
        cv2.circle(
            annotated,
            (int(round(point.x)), int(round(point.y))),
            point_radius,
            color,
            thickness,
            cv2.LINE_AA,
        )

    for rail_index, rail in enumerate(result.rails):
        segment = line_image_segment(rail, width, height)
        if segment is not None:
            cv2.line(
                annotated,
                segment[0],
                segment[1],
                RAIL_COLORS[rail_index],
                thickness,
                cv2.LINE_AA,
            )

    if len(result.corners) == 4:
        polygon = np.asarray(result.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            annotated, [polygon], True, CORNER_COLOR, thickness, cv2.LINE_AA
        )
        for index, corner in enumerate(result.corners):
            position = tuple(int(round(value)) for value in corner)
            cv2.circle(
                annotated,
                position,
                point_radius + 3,
                CORNER_COLOR,
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                str(index),
                (position[0] + 8, position[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, min(width, height) / 1000),
                CORNER_COLOR,
                thickness,
                cv2.LINE_AA,
            )

    supports = ",".join(str(len(rail.inlier_indices)) for rail in result.rails)
    summary = (
        f"rail fit | mode={mode} valid={result.valid} dots={len(result.points)} "
        f"rails={len(result.rails)} support=[{supports}] "
        f"threshold={result.distance_threshold_px:.1f}px"
    )
    warning_text = "; ".join(result.warnings) if result.warnings else "warnings=none"
    scale = max(0.55, min(width, height) / 1100)
    text_thickness = max(1, round(min(width, height) / 600))
    text_height = cv2.getTextSize(
        summary, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness
    )[0][1]
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (width - 1, text_height * 3 + 20),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.80, annotated, 0.20, 0, annotated)
    cv2.putText(
        annotated,
        summary,
        (8, text_height + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        warning_text[:180],
        (8, text_height * 2 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        text_thickness,
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
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            bool(row["valid"]),
            int(row["rail_count"]),
            int(row["assigned_dot_count"]),
        ),
    )
    tile_width, tile_height = 480, 300
    tiles: list[np.ndarray] = []
    for row in ordered_rows[:24]:
        source = cv2.imread(str(output_dir / str(row["overlay"])))
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
        tile = np.full((tile_height, tile_width, 3), 25, dtype=np.uint8)
        x_offset = (tile_width - resized.shape[1]) // 2
        y_offset = (tile_height - resized.shape[0]) // 2
        tile[
            y_offset : y_offset + resized.shape[0],
            x_offset : x_offset + resized.shape[1],
        ] = resized
        tiles.append(tile)
    if not tiles:
        return
    columns = 3
    blank = np.full_like(tiles[0], 25)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    contact_rows = [
        np.hstack(tiles[index : index + columns])
        for index in range(0, len(tiles), columns)
    ]
    cv2.imwrite(str(output_path), np.vstack(contact_rows))


def markdown_report(
    args: argparse.Namespace,
    settings: dict[str, object],
    rows: Sequence[dict[str, object]],
) -> str:
    valid_count = sum(bool(row["valid"]) for row in rows)
    lines = [
        "# Four-rail fitting report",
        "",
        f"- Input mode: `{args.mode}`",
        f"- Images: `{len(rows)}`",
        f"- Structurally valid fits: `{valid_count}/{len(rows)}`",
        "- Held-out detector test split used: **no**",
        "- Matched-view geometry evaluation set used: **no**",
        "",
        "A structurally valid result means four supported lines produced a convex,",
        "plausibly sized quadrilateral. It does not yet prove correct semantic rail",
        "ordering or homography accuracy; inspect the overlays.",
        "",
        "## Settings",
        "",
        f"- Rail distance threshold ratio: `{settings['rail_distance_threshold_ratio']}`",
        f"- Minimum inliers per rail: `{settings['min_inliers_per_rail']}`",
        f"- Minimum total dots: `{settings['min_total_points']}`",
    ]
    if args.mode == "yolo":
        confidence = (
            args.confidence
            if args.confidence is not None
            else float(settings["prediction_confidence"])
        )
        lines.append(f"- YOLO confidence: `{confidence:.2f}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Image | Dots | Rails | Supports | Area fraction | Valid | Warnings |",
            "|---|---:|---:|---|---:|---|---|",
            *(
                f"| {row['image']} | {row['dot_count']} | {row['rail_count']} | "
                f"{row['rail_supports']} | {row['area_fraction']} | "
                f"{row['valid']} | {row['warnings'] or '-'} |"
                for row in rows
            ),
            "",
            "Open `contact_sheet.jpg` first, then inspect individual overlays and",
            "JSON files for any suspicious result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    config = rail_config(settings)
    image_paths = find_images(args.source)
    args.output.mkdir(parents=True, exist_ok=True)

    confidence: float | None = None
    rows: list[dict[str, object]] = []

    def process_image(
        image_path: Path,
        image: np.ndarray,
        points: list[PointObservation],
    ) -> None:
        result = fit_rails(
            points,
            image_size=(image.shape[1], image.shape[0]),
            config=config,
        )
        overlay_name = f"{image_path.stem}_rails.jpg"
        json_name = f"{image_path.stem}_rails.json"
        overlay = render_result(image, result, args.mode)
        cv2.imwrite(str(args.output / overlay_name), overlay)
        payload = {
            "image": str(image_path),
            "dot_source": args.mode,
            "prediction_confidence": confidence,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "geometry": result.as_dict(),
        }
        (args.output / json_name).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        assigned_count = sum(len(rail.inlier_indices) for rail in result.rails)
        rows.append(
            {
                "image": image_path.name,
                "dot_source": args.mode,
                "dot_count": len(points),
                "rail_count": len(result.rails),
                "rail_supports": "/".join(
                    str(len(rail.inlier_indices)) for rail in result.rails
                ),
                "assigned_dot_count": assigned_count,
                "outlier_count": len(result.outlier_indices),
                "area_fraction": (
                    f"{result.quadrilateral_area_fraction:.4f}"
                    if result.quadrilateral_area_fraction is not None
                    else ""
                ),
                "valid": result.valid,
                "warnings": ";".join(result.warnings),
                "overlay": overlay_name,
                "json": json_name,
            }
        )

    if args.mode == "labels":
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            points = load_labelled_dots(
                image_path,
                (image.shape[1], image.shape[0]),
                args.labels_dir,
                args.dot_class_id,
            )
            process_image(image_path, image, points)
    else:
        from ultralytics import YOLO

        confidence = (
            args.confidence
            if args.confidence is not None
            else float(settings["prediction_confidence"])
        )
        imgsz = args.imgsz if args.imgsz is not None else int(settings["imgsz"])
        model = YOLO(str(args.weights))
        dot_class_id = yolo_dot_class_id(model)
        predictions = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=imgsz,
            device=args.device,
            batch=args.batch,
            conf=confidence,
            iou=float(settings["prediction_nms_iou"]),
            save=False,
            verbose=False,
            stream=True,
        )
        processed_count = 0
        for image_path, prediction in zip(image_paths, predictions):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            process_image(
                image_path,
                image,
                prediction_points(prediction, dot_class_id),
            )
            processed_count += 1
        if processed_count != len(image_paths):
            raise RuntimeError(
                f"Expected {len(image_paths)} predictions, got {processed_count}"
            )

    write_csv(args.output / "summary.csv", rows)
    create_contact_sheet(args.output, rows, args.output / "contact_sheet.jpg")
    report_path = args.output / "RAIL_FITTING_REPORT.md"
    report_path.write_text(
        markdown_report(args, settings, rows), encoding="utf-8"
    )
    valid_count = sum(bool(row["valid"]) for row in rows)
    print(
        f"mode={args.mode} images={len(rows)} structurally_valid={valid_count}/{len(rows)}"
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
