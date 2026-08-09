from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from dataset_common import (
    CLASS_NAMES,
    class_counts,
    label_path_for_image,
    list_images,
    parse_label,
    source_id_from_filename,
    verify_image,
)


def audit_dataset(dataset_root: Path, allow_polygon_repair: bool) -> dict:
    report: dict = {
        "dataset": str(dataset_root.resolve()),
        "classes": CLASS_NAMES,
        "splits": {},
        "totals": {
            "images": 0,
            "labels": 0,
            "boxes": 0,
            "repairable_polygon_rows": 0,
            "class_counts": {name: 0 for name in CLASS_NAMES},
        },
        "warnings": [],
        "errors": [],
    }

    for split in ("train", "valid", "test", "geometry_eval"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.exists() and not labels_dir.exists():
            continue

        images = list_images(images_dir)
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        split_counts: Counter[int] = Counter()
        dimensions: Counter[str] = Counter()
        split_boxes = 0
        split_repairs = 0
        source_kinds: Counter[str] = Counter()

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in label_files}
        for stem in sorted(image_stems - label_stems):
            report["errors"].append(f"{split}: missing label for {stem}")
        for stem in sorted(label_stems - image_stems):
            report["errors"].append(f"{split}: label without image for {stem}")

        for image_path in images:
            source_id = source_id_from_filename(image_path.name)
            if source_id[-1:].lower() in {"a", "f", "t"} and source_id[:-1].isdigit():
                source_kinds["multi_view"] += 1
            elif source_id.isdigit():
                source_kinds["main"] += 1
            else:
                source_kinds["other"] += 1

            try:
                width, height = verify_image(image_path)
                dimensions[f"{width}x{height}"] += 1
            except Exception as exc:  # Pillow raises several format-specific errors.
                report["errors"].append(f"{split}: unreadable image {image_path.name}: {exc}")

            label_path = label_path_for_image(image_path, labels_dir)
            if not label_path.exists():
                continue
            try:
                parsed = parse_label(label_path, allow_polygon_repair=allow_polygon_repair)
            except ValueError as exc:
                report["errors"].append(str(exc))
                continue
            split_boxes += len(parsed.boxes)
            split_repairs += parsed.repair_count
            split_counts.update(class_counts(parsed.boxes))
            report["warnings"].extend(parsed.warnings)

        report["splits"][split] = {
            "images": len(images),
            "labels": len(label_files),
            "boxes": split_boxes,
            "repairable_polygon_rows": split_repairs,
            "dimensions": dict(sorted(dimensions.items())),
            "source_kinds": dict(sorted(source_kinds.items())),
            "class_counts": {CLASS_NAMES[index]: split_counts[index] for index in range(len(CLASS_NAMES))},
        }
        report["totals"]["images"] += len(images)
        report["totals"]["labels"] += len(label_files)
        report["totals"]["boxes"] += split_boxes
        report["totals"]["repairable_polygon_rows"] += split_repairs
        for index, name in enumerate(CLASS_NAMES):
            report["totals"]["class_counts"][name] += split_counts[index]

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a YOLO billiards dataset.")
    parser.add_argument("dataset", type=Path, help="Dataset root containing split directories")
    parser.add_argument("--json-output", type=Path, help="Optional report path")
    parser.add_argument(
        "--allow-polygon-repair",
        action="store_true",
        help="Treat valid polygon rows as repairable warnings instead of hard errors",
    )
    args = parser.parse_args()

    report = audit_dataset(args.dataset, allow_polygon_repair=args.allow_polygon_repair)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

