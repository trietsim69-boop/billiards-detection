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
from dataclasses import fields
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

from analyze_detector_errors import (  # noqa: E402
    IMAGE_SUFFIXES,
    create_contact_sheet,
    draw_banner,
    load_ground_truth,
    result_to_predictions,
    write_csv,
)
from billiards.geometry import (  # noqa: E402
    PointObservation,
    RailFitConfig,
    RailFitResult,
    RailLine,
    fit_rails,
)


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
        default=Path("outputs/detection/baseline-yolo11n-960/weights/best.pt"),
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
    return RailFitConfig(**{field.name: settings[field.name] for field in fields(RailFitConfig)})


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


def load_labelled_dots(
    image_path: Path,
    image_size: tuple[int, int],
    labels_dir: Path | None,
    dot_class_id: int,
) -> list[PointObservation]:
    return [
        PointObservation(*item.center, 1.0, index)
        for index, item in enumerate(load_ground_truth(image_path, *image_size, labels_dir))
        if item.class_id == dot_class_id
    ]


def yolo_dot_class_id(model: object) -> int:
    matches = [index for index, name in model.names.items() if name == "Dot"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Dot class, found {matches}")
    return int(matches[0])


def prediction_points(result: object, dot_class_id: int) -> list[PointObservation]:
    return [
        PointObservation(*item.center, item.confidence, index)
        for index, item in enumerate(result_to_predictions(result))
        if item.class_id == dot_class_id
    ]


def line_image_segment(
    line: RailLine, width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # (-a*c, -b*c) is the line's closest point to the origin; extend it past the image.
    x, y, far = -line.a * line.c, -line.b * line.c, 4 * (width + height)
    inside, start, end = cv2.clipLine(
        (0, 0, width, height),
        (round(x + far * line.b), round(y - far * line.a)),
        (round(x - far * line.b), round(y + far * line.a)),
    )
    return (start, end) if inside else None


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
    draw_banner(
        annotated,
        [
            f"rail fit | mode={mode} valid={result.valid} dots={len(result.points)} "
            f"rails={len(result.rails)} support=[{supports}] "
            f"threshold={result.distance_threshold_px:.1f}px",
            ("; ".join(result.warnings) or "warnings=none")[:180],
        ],
    )
    return annotated


def rail_contact_sheet(
    output_dir: Path, rows: Sequence[dict[str, object]], output_path: Path
) -> None:
    """Contact sheet of up to 24 overlays, failed and weakly supported fits first."""
    ordered = sorted(
        rows,
        key=lambda row: (bool(row["valid"]), int(row["rail_count"]), int(row["assigned_dot_count"])),
    )
    create_contact_sheet([output_dir / str(row["overlay"]) for row in ordered[:24]], output_path)


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
        f"- Rail distance threshold ratio: `{settings['distance_threshold_ratio']}`",
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
        for image_path, prediction in zip(image_paths, predictions, strict=True):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            process_image(
                image_path,
                image,
                prediction_points(prediction, dot_class_id),
            )

    write_csv(args.output / "summary.csv", rows)
    rail_contact_sheet(args.output, rows, args.output / "contact_sheet.jpg")
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
