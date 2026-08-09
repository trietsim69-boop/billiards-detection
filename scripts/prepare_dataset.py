from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from dataset_common import (
    CLASS_NAMES,
    Box,
    class_counts,
    format_boxes,
    hamming_distance,
    image_dhash,
    label_path_for_image,
    list_images,
    parse_label,
    sha256_file,
    source_id_from_filename,
    verify_image,
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def aggregate_counts(items: list[dict]) -> Counter[int]:
    total: Counter[int] = Counter()
    for item in items:
        total.update(item["class_counts"])
    return total


def make_groups(items: list[dict], distance_threshold: int) -> list[dict]:
    union_find = UnionFind(len(items))
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if hamming_distance(items[left]["dhash"], items[right]["dhash"]) <= distance_threshold:
                union_find.union(left, right)

    grouped: dict[int, list[dict]] = {}
    for index, item in enumerate(items):
        grouped.setdefault(union_find.find(index), []).append(item)

    groups: list[dict] = []
    for members in grouped.values():
        members.sort(key=lambda value: int(value["source_id"]))
        groups.append(
            {
                "members": members,
                "size": len(members),
                "class_counts": aggregate_counts(members),
                "minimum_source": min(int(member["source_id"]) for member in members),
            }
        )
    groups.sort(key=lambda group: group["minimum_source"])
    for index, group in enumerate(groups, 1):
        group["group_id"] = f"visual-{index:03d}"
    return groups


def take_groups(order: list[dict], target_size: int) -> tuple[list[dict], list[dict]] | None:
    selected: list[dict] = []
    remaining_groups: list[dict] = []
    remaining_size = target_size
    for group in order:
        if group["size"] <= remaining_size:
            selected.append(group)
            remaining_size -= group["size"]
        else:
            remaining_groups.append(group)
    if remaining_size:
        return None
    selected_ids = {id(group) for group in selected}
    remaining_groups.extend(group for group in order if id(group) not in selected_ids and group not in remaining_groups)
    return selected, remaining_groups


def distribution_score(
    split_groups: list[dict],
    split_size: int,
    global_counts: Counter[int],
    total_images: int,
) -> float:
    actual: Counter[int] = Counter()
    for group in split_groups:
        actual.update(group["class_counts"])
    score = 0.0
    for class_id in range(len(CLASS_NAMES)):
        expected = global_counts[class_id] * split_size / total_images
        score += abs(actual[class_id] - expected) / max(expected, 1.0)
    return score


def choose_splits(groups: list[dict], seed: int, trials: int = 20_000) -> dict[str, list[dict]]:
    total_images = sum(group["size"] for group in groups)
    if total_images != 195:
        raise ValueError(f"Expected 195 main images, found {total_images}")

    global_counts: Counter[int] = Counter()
    for group in groups:
        global_counts.update(group["class_counts"])

    random_generator = random.Random(seed)
    best: tuple[float, tuple, list[dict], list[dict], list[dict]] | None = None
    for _ in range(trials):
        order = groups.copy()
        random_generator.shuffle(order)
        test_result = take_groups(order, 20)
        if test_result is None:
            continue
        test_groups, after_test = test_result
        random_generator.shuffle(after_test)
        valid_result = take_groups(after_test, 20)
        if valid_result is None:
            continue
        valid_groups, train_groups = valid_result
        if sum(group["size"] for group in train_groups) != 155:
            continue

        score = distribution_score(test_groups, 20, global_counts, total_images)
        score += distribution_score(valid_groups, 20, global_counts, total_images)
        signature = (
            tuple(sorted(group["minimum_source"] for group in test_groups)),
            tuple(sorted(group["minimum_source"] for group in valid_groups)),
        )
        candidate = (score, signature, train_groups, valid_groups, test_groups)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to produce exact grouped 155/20/20 splits")
    return {"train": best[2], "valid": best[3], "test": best[4]}


def load_item(image_path: Path, labels_dir: Path) -> dict:
    label_path = label_path_for_image(image_path, labels_dir)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {image_path.name}")
    verify_image(image_path)
    parsed = parse_label(label_path, allow_polygon_repair=True)
    return {
        "source_id": source_id_from_filename(image_path.name),
        "image": image_path,
        "label": label_path,
        "parsed": parsed,
        "class_counts": class_counts(parsed.boxes),
        "dhash": image_dhash(image_path),
    }


def copy_item(item: dict, build_root: Path, split: str, group_id: str) -> dict:
    images_dir = build_root / split / "images"
    labels_dir = build_root / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    destination_image = images_dir / item["image"].name
    destination_label = labels_dir / item["label"].name
    shutil.copy2(item["image"], destination_image)
    destination_label.write_text(format_boxes(item["parsed"].boxes), encoding="utf-8")

    return {
        "source_id": item["source_id"],
        "scene_group": group_id,
        "split": split,
        "source_image": str(item["image"].resolve()),
        "source_label": str(item["label"].resolve()),
        "image_filename": destination_image.name,
        "label_filename": destination_label.name,
        "image_sha256": sha256_file(item["image"]),
        "label_sha256": sha256_file(item["label"]),
        "dhash": f"{item['dhash']:016x}",
        "repair_count": item["parsed"].repair_count,
    }


def summarize_output(build_root: Path, manifest_rows: list[dict], distance_threshold: int, seed: int) -> dict:
    summary = {
        "seed": seed,
        "near_duplicate_hamming_threshold": distance_threshold,
        "splits": {},
        "repairs": [row for row in manifest_rows if row["repair_count"]],
    }
    for split in ("train", "valid", "test", "geometry_eval"):
        labels_dir = build_root / split / "labels"
        images = list_images(build_root / split / "images")
        total_counts: Counter[int] = Counter()
        boxes = 0
        for image in images:
            parsed = parse_label(label_path_for_image(image, labels_dir), allow_polygon_repair=False)
            total_counts.update(class_counts(parsed.boxes))
            boxes += len(parsed.boxes)
        summary["splits"][split] = {
            "images": len(images),
            "labels": len(list(labels_dir.glob("*.txt"))),
            "boxes": boxes,
            "class_counts": {CLASS_NAMES[index]: total_counts[index] for index in range(len(CLASS_NAMES))},
        }
    return summary


def write_metadata(build_root: Path, manifest_rows: list[dict], summary: dict) -> None:
    yaml_text = """train: train/images
val: valid/images
test: test/images

nc: 5
names: ['Black', 'Cue', 'Dot', 'Solid', 'Striped']
"""
    (build_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    fieldnames = [
        "source_id",
        "scene_group",
        "split",
        "source_image",
        "source_label",
        "image_filename",
        "label_filename",
        "image_sha256",
        "label_sha256",
        "dhash",
        "repair_count",
    ]
    with (build_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda row: (row["split"], row["scene_group"], row["source_id"])))

    (build_root / "audit_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    readme = """# Generated Pix2Pockets v3 split

This directory is generated from the immutable Roboflow export in
`8-Ball Pool.v3i.yolov11`.

- The 195 main images are split into 155 train, 20 validation, and 20 test images.
- Conservative dHash grouping keeps near-duplicate images in one split.
- The 52 `a`, `f`, and `t` images are reserved for geometry evaluation and grouped by their 25 situation identifiers.
- Valid segmentation polygon rows are converted to tight YOLO detection boxes in this processed copy.
- `split_manifest.csv` records provenance, hashes, grouping, and repairs.

Do not edit this directory manually. Regenerate it with `scripts/prepare_dataset.py`.
"""
    (build_root / "README.md").write_text(readme, encoding="utf-8")


def prepare(source_root: Path, output_root: Path, seed: int, distance_threshold: int) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_root}")

    images_dir = source_root / "train" / "images"
    labels_dir = source_root / "train" / "labels"
    images = list_images(images_dir)
    if len(images) != 247:
        raise ValueError(f"Expected 247 source images, found {len(images)}")

    main_items: list[dict] = []
    geometry_items: list[dict] = []
    for image in images:
        item = load_item(image, labels_dir)
        source_id = item["source_id"]
        if re.fullmatch(r"\d+", source_id):
            main_items.append(item)
        elif re.fullmatch(r"\d+[aft]", source_id, re.IGNORECASE):
            geometry_items.append(item)
        else:
            raise ValueError(f"Unexpected source id: {source_id}")

    if len(main_items) != 195 or len(geometry_items) != 52:
        raise ValueError(f"Expected 195 main and 52 geometry images, found {len(main_items)} and {len(geometry_items)}")

    groups = make_groups(main_items, distance_threshold)
    assignments = choose_splits(groups, seed)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    manifest_rows: list[dict] = []
    try:
        for split, split_groups in assignments.items():
            for group in split_groups:
                for item in group["members"]:
                    manifest_rows.append(copy_item(item, temporary_root, split, group["group_id"]))

        for item in geometry_items:
            situation = re.match(r"\d+", item["source_id"]).group()
            manifest_rows.append(copy_item(item, temporary_root, "geometry_eval", f"situation-{int(situation):02d}"))

        summary = summarize_output(temporary_root, manifest_rows, distance_threshold, seed)
        expected_counts = {"train": 155, "valid": 20, "test": 20, "geometry_eval": 52}
        actual_counts = {split: details["images"] for split, details in summary["splits"].items()}
        if actual_counts != expected_counts:
            raise RuntimeError(f"Unexpected output counts: {actual_counts}")

        write_metadata(temporary_root, manifest_rows, summary)
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic Pix2Pockets detection and geometry splits.")
    parser.add_argument("source", type=Path, help="Raw Roboflow dataset root")
    parser.add_argument("output", type=Path, help="New processed dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    args = parser.parse_args()

    summary = prepare(args.source, args.output, args.seed, args.near_duplicate_distance)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

