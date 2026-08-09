from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

from PIL import Image


CLASS_NAMES = ["Black", "Cue", "Dot", "Solid", "Striped"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class ParsedLabel:
    boxes: tuple[Box, ...]
    repair_count: int
    warnings: tuple[str, ...]


def source_id_from_filename(filename: str) -> str:
    """Recover the Roboflow source name from a generated filename."""
    match = re.match(r"(.+?)_(?:png|jpe?g)\.rf\.[^.]+\.[^.]+$", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return Path(filename).stem


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _validate_box(box: Box, context: str) -> None:
    if not 0 <= box.class_id < len(CLASS_NAMES):
        raise ValueError(f"{context}: invalid class id {box.class_id}")
    values = (box.x, box.y, box.width, box.height)
    if not all(value == value for value in values):
        raise ValueError(f"{context}: NaN coordinate")
    if not (0 <= box.x <= 1 and 0 <= box.y <= 1):
        raise ValueError(f"{context}: box centre outside normalized image")
    if not (0 < box.width <= 1 and 0 < box.height <= 1):
        raise ValueError(f"{context}: invalid box dimensions")
    if box.x - box.width / 2 < -1e-6 or box.x + box.width / 2 > 1 + 1e-6:
        raise ValueError(f"{context}: box crosses horizontal image bounds")
    if box.y - box.height / 2 < -1e-6 or box.y + box.height / 2 > 1 + 1e-6:
        raise ValueError(f"{context}: box crosses vertical image bounds")


def parse_label(path: Path, allow_polygon_repair: bool) -> ParsedLabel:
    boxes: list[Box] = []
    warnings: list[str] = []
    repair_count = 0

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        context = f"{path.name}:{line_number}"
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{context}: non-numeric annotation") from exc

        if len(parts) == 5:
            box = Box(class_id, *coordinates)
        elif len(parts) >= 7 and len(coordinates) % 2 == 0:
            if not allow_polygon_repair:
                raise ValueError(f"{context}: segmentation polygon in detection dataset")
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            if not all(0 <= value <= 1 for value in xs + ys):
                raise ValueError(f"{context}: polygon point outside normalized image")
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            box = Box(
                class_id=class_id,
                x=(x_min + x_max) / 2,
                y=(y_min + y_max) / 2,
                width=x_max - x_min,
                height=y_max - y_min,
            )
            repair_count += 1
            warnings.append(f"{context}: polygon converted to bounding box")
        else:
            raise ValueError(f"{context}: expected 5 fields or a valid polygon row")

        _validate_box(box, context)
        boxes.append(box)

    return ParsedLabel(tuple(boxes), repair_count, tuple(warnings))


def format_boxes(boxes: tuple[Box, ...]) -> str:
    lines = [
        f"{box.class_id} {box.x:.10f} {box.y:.10f} {box.width:.10f} {box.height:.10f}"
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def class_counts(boxes: tuple[Box, ...]) -> Counter[int]:
    return Counter(box.class_id for box in boxes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(path: Path) -> int:
    """Return a 64-bit difference hash for conservative near-duplicate grouping."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())

    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return size


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()

