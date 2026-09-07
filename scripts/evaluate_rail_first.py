"""Compare frozen YOLO weights on full validation frames and new rail strips.

Experimental detection comparison only: the cloth locator is unreviewed and
the current rail/lattice implementation is evaluated as-is. Labels are used
only after predictions, for scoring. Test and geometry holdout are not read.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from billiards.geometry import fit_rails
from billiards.rail_regions import RailRegionConfig, extract_rail_regions
from billiards.rail_sights import RailSightCandidate, RailSightConfig, detect_rail_sights
from billiards.table_localization import TableLocalizationConfig, localize_table
from scripts.analyze_detector_errors import Detection
from scripts.analyze_dot_centers import match_centers, evaluate_setting, write_csv
from scripts.fit_table_rails import load_labelled_dots, prediction_points, yolo_dot_class_id
from ultralytics import YOLO


def as_detection(point):
    return Detection(2, (point.x - 1, point.y - 1, point.x + 1, point.y + 1), point.confidence)


def metric_row(gt, predicted, gain, tolerance):
    matches = match_centers(gt, predicted, tolerance, gain)
    tp = len(matches)
    fp, fn = len(predicted) - tp, len(gt) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * precision * recall / max(1e-12, precision + recall))


def draw_panel(image, gt, predictions, gain, tolerance, title):
    matches = match_centers(gt, predictions, tolerance, gain)
    hit_gt = {m.ground_truth_index for m in matches}
    hit_pred = {m.prediction_index for m in matches}
    panel = image.copy()
    for i, point in enumerate(gt):
        if i not in hit_gt:
            x1, y1, x2, y2 = point.box_xyxy
            cv2.circle(panel, (round((x1+x2)/2), round((y1+y2)/2)), 9, (30, 30, 240), 2)
    for i, point in enumerate(predictions):
        x1, y1, x2, y2 = point.box_xyxy
        cv2.drawMarker(panel, (round((x1+x2)/2), round((y1+y2)/2)),
                       (50, 230, 50) if i in hit_pred else (0, 170, 255), cv2.MARKER_CROSS, 12, 2)
    panel = cv2.resize(panel, (640, round(640 * image.shape[0] / image.shape[1])))
    banner = np.full((45, 640, 3), 25, dtype=np.uint8)
    metrics = metric_row(gt, predictions, gain, tolerance)
    cv2.putText(banner, f"{title}: TP {metrics['tp']} FP {metrics['fp']} FN {metrics['fn']}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return np.vstack((banner, panel))


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs/detection/baseline-yolo11n-960/weights/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    source = ROOT / "data/processed/pix2pockets_v3/valid/images"
    paths = sorted(source.glob("*.jpg"))
    if len(paths) != 20:
        raise RuntimeError(f"Expected all 20 validation images, found {len(paths)}")
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory to preserve prior runs")
    args.output.mkdir(parents=True)
    for name in ("overlays", "rail_strips", "localization"):
        (args.output / name).mkdir()
    settings = yaml.safe_load((ROOT / "configs/rail_first.yaml").read_text())
    region_config = RailRegionConfig(**{f.name: settings[f.name] for f in fields(RailRegionConfig) if f.name in settings})
    sight_config = RailSightConfig(**{f.name: settings[f.name] for f in fields(RailSightConfig) if f.name in settings})
    locator_config = TableLocalizationConfig()
    configs = dict(imgsz=960, min_confidence=0.01, nms_iou=0.70,
                   thresholds=[0.05, 0.25, 0.50], tolerances=[4.0, 8.0, 12.0],
                   region=asdict(region_config), sight=asdict(sight_config), locator=asdict(locator_config),
                   weights=str(args.weights.resolve()), weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                   split="valid", test_split_used=False, model_training=False)
    (args.output / "config.yaml").write_text(yaml.safe_dump(configs), encoding="utf-8")
    # Snapshot inference code for reproducibility, including uncommitted work.
    code_files = [Path(__file__), *sorted((ROOT / "src/billiards").glob("*.py")), ROOT / "scripts/analyze_dot_centers.py"]
    (args.output / "source_snapshot.json").write_text(json.dumps({str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in code_files}, indent=2), encoding="utf-8")
    write_csv(args.output / "manifest.csv", [dict(image=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    model = YOLO(str(args.weights))
    dot_id = yolo_dot_class_id(model)
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()

    def infer(image):
        return model.predict(image, imgsz=960, conf=0.01, iou=0.70, device=args.device,
                             batch=1, verbose=False, save=False)[0]

    variant_names = ("A_full_frame", "B_crop_yolo", "C_classical_filtered", "D_crop_hybrid", "E_full_frame_hybrid")
    stored = {conf: {name: [] for name in variant_names} for conf in configs["thresholds"]}
    ground_truth, gains, rows, timings, tiles = [], [], [], [], []
    localized = 0
    raw_handle = (args.output / "candidates.jsonl").open("w", encoding="utf-8")
    try:
        for index, path in enumerate(paths):
            start = time.perf_counter()
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Unreadable image: {path}")
            height, width = image.shape[:2]
            gain = 960 / max(width, height)
            full = prediction_points(infer(image), dot_id)
            full_done = time.perf_counter()
            table = localize_table(image, locator_config)
            crop, crop_records, strips = [], [], []
            local_image = image.copy()
            if table.valid:
                localized += 1
                cv2.polylines(local_image, [np.asarray(table.bed_corners_xy, dtype=np.int32)], True, (0,255,255), 3)
                for region in extract_rail_regions(image, table, region_config):
                    save_image(args.output / "rail_strips" / f"{path.stem}_{region.side}.jpg", region.strip_image)
                    strip_predictions = prediction_points(infer(region.strip_image), dot_id)
                    crop_records.append(dict(side=region.side, source_polygon=region.source_polygon_xy,
                                             coverage=region.valid_pixel_coverage, warnings=region.warnings,
                                             source_to_strip=region.source_to_strip.tolist(), strip_to_source=region.strip_to_source.tolist(),
                                             predictions=[asdict(p) for p in strip_predictions]))
                    for point in strip_predictions:
                        x, y = region.strip_to_source_point((point.x, point.y))
                        if 0 <= x < width and 0 <= y < height:
                            crop.append(RailSightCandidate(x, y, point.confidence, "crop_yolo", region.side))
            save_image(args.output / "localization" / path.name, local_image)
            inference_done = time.perf_counter()
            variants_by_conf = {}
            diagnostic = {}
            for conf in configs["thresholds"]:
                full_selected = [p for p in full if p.confidence >= conf]
                crop_selected = [p for p in crop if p.confidence >= conf]
                variants = {name: [] for name in variant_names}
                variants["A_full_frame"] = full_selected
                variants["B_crop_yolo"] = crop_selected
                if table.valid:
                    for name, learned in (("C_classical_filtered", []), ("D_crop_hybrid", crop_selected), ("E_full_frame_hybrid", full_selected)):
                        result = detect_rail_sights(image, table, learned, sight_config, region_config)
                        variants[name] = list(result.accepted_centres)
                        diagnostic[f"{name}@{conf}"] = result.as_dict()
                variants_by_conf[conf] = variants
            # Ground truth is read strictly after all inference and filtering.
            gt_points = load_labelled_dots(path, (width, height), None, dot_id)
            gt = [as_detection(p) for p in gt_points]
            ground_truth.append(gt)
            gains.append(gain)
            for conf, variants in variants_by_conf.items():
                for name, points in variants.items():
                    predicted = [as_detection(p) for p in points]
                    stored[conf][name].append(predicted)
                    # Keep existing fit_rails acceptance gates unchanged.
                    geometry = fit_rails([p.as_point() if isinstance(p, RailSightCandidate) else p for p in points], (width, height))
                    for tolerance in configs["tolerances"]:
                        rows.append(dict(image=path.name, variant=name, confidence=conf, tolerance=tolerance,
                                         localized=table.valid, geometry_valid=geometry.valid,
                                         **metric_row(gt, predicted, gain, tolerance)))
            shown = variants_by_conf[0.25]
            panels = [draw_panel(image, gt, [as_detection(p) for p in shown[name]], gain, 8.0, name) for name in ("A_full_frame", "B_crop_yolo", "D_crop_hybrid")]
            comparison = np.hstack(panels)
            save_image(args.output / "overlays" / path.name, comparison)
            tiles.append(cv2.resize(comparison, (960, round(comparison.shape[0] / 2))))
            elapsed = time.perf_counter() - start
            timings.append(dict(image=path.name, full_frame_seconds=full_done-start,
                                localization_and_crop_seconds=inference_done-full_done, total_seconds=elapsed))
            raw_handle.write(json.dumps(dict(image=path.name, table=table.as_dict(), full_frame=[asdict(p) for p in full],
                                             crop_records=crop_records, postprocessing=diagnostic)) + "\n")
            raw_handle.flush()
            print(f"{index+1}/20 {path.name}: localized={table.valid} full={len(full)} crop={len(crop)} ({elapsed:.1f}s)", flush=True)
    finally:
        raw_handle.close()
    summary_rows = []
    for conf, variants in stored.items():
        for name, predictions in variants.items():
            for tolerance in configs["tolerances"]:
                # Filtering already applied; classical scores aren't YOLO probabilities.
                row = evaluate_setting(ground_truth, predictions, gains, 0.0, tolerance)
                row.update(variant=name, confidence=conf)
                row["geometry_valid_images"] = sum(r["geometry_valid"] for r in rows if r["variant"] == name and r["confidence"] == conf and r["tolerance"] == tolerance)
                summary_rows.append(row)
    summary = dict(image_count=len(paths), ground_truth_dots=sum(map(len, ground_truth)), localized_images=localized,
                   metrics=summary_rows, config=configs,
                   peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if args.device != "cpu" else None)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output / "per_image.csv", rows)
    write_csv(args.output / "timings.csv", timings)
    write_csv(args.output / "metrics.csv", summary_rows)
    save_image(args.output / "contact_sheet.jpg", np.vstack(tiles))
    report = ["# Rail-first inference experiment", "", f"All {len(paths)} validation images; {summary['ground_truth_dots']} labelled dots. Locator proposals: {localized}/20.", "",
              "Existing YOLO11n 960 checkpoint; no retraining. Test and geometry holdout untouched.",
              "Matching is one-to-one centre distance at 960-pixel scale. All images, including locator failures, remain in denominators.", "",
              "## Matched-confidence comparison (8px at 960)", "",
              "| Variant | YOLO conf | TP | FP | FN | Precision | Recall | F1 | Structurally valid fits |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        if row["tolerance_model_px"] == 8.0:
            report.append(f"| {row['variant']} | {row['confidence']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['geometry_valid_images']}/20 |")
    report += ["", "## Interpretation limits", "",
               "A: full-frame YOLO; B: mapped raw crop YOLO (overlap duplicates count as false positives); C: classical proposals plus existing lattice; D: crop YOLO plus classical/lattice; E: full-frame YOLO plus classical/lattice.",
               "Classical scores have no calibrated equivalence to YOLO confidence; the confidence column applies only to learned inputs.",
               "The provisional locator uses image colour and quadrilateral shape. It lacks reviewed bed annotations; long-side roles use projected edge lengths.",
               "The prior rail implementation uses offset parallelogram bands (not complete projective rail rectification), fixed lattice positions, and image-relative deduplication. These are evaluated without tuning them to these results.",
               "Structural fit validity does not establish correct geometry. No bed-plane or ball-projection accuracy claim is supported by this experiment.",
               "Runtime includes startup for the first frame; timings.csv splits full-frame from localization/crop inference. No runtime optimization performed.",
               "Overlays show confidence 0.25, tolerance 8px: green=matched, orange=false positive, red=missed. Full candidate traces and source snapshots are saved."]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "localized": localized, "metrics_at_025_8px": [r for r in summary_rows if r["confidence"] == .25 and r["tolerance_model_px"] == 8.0]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
