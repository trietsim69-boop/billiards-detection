"""Prepare, train and evaluate YOLO on original frames plus overlapping tiles.

Run actions separately so the unchanged checkpoint can be evaluated before any
training. Only train/valid are accessed. Fresh output directories are required.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Process-local limits: reduce memory/CPU contention on the training laptop.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from billiards.crop_detection import BoxDetection, CropBox, crop_labels, overlapping_tiles, supplement_detections
from scripts.analyze_dot_centers import evaluate_setting, write_csv
from scripts.analyze_detector_errors import Detection, load_ground_truth, result_to_predictions
from scripts.evaluate_rail_first import draw_panel, save_image
from ultralytics import YOLO

SOURCE = ROOT / "data/processed/pix2pockets_v3"
DATASET = ROOT / "outputs/datasets/aspect-crops-v1"
TRAIN_CONFIG = ROOT / "configs/train_yolo11n_crop_finetune.yaml"
BASELINE = ROOT / "outputs/detection/baseline-yolo11n-960/weights/best.pt"
CONFIDENCES = (0.05, 0.25, 0.5, 0.75)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_labels(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Bad label row: {path}")
        row = (int(values[0]), *map(float, values[1:]))
        if not 0 <= row[0] < 5 or not all(np.isfinite(row)) or not all(0 <= v <= 1 for v in row[1:]):
            raise ValueError(f"Invalid class/coordinates: {path}")
        rows.append(row)
    return rows


def prepare():
    names = yaml.safe_load((SOURCE / "data.yaml").read_text())["names"]
    source_paths = {split: sorted((SOURCE / split / "images").glob("*.jpg")) for split in ("train", "valid")}
    if [len(source_paths[s]) for s in ("train", "valid")] != [155, 20]:
        raise RuntimeError("Expected the complete 155 training and 20 validation frames")
    # Reject exact cross-split source duplicates before copying/cropping.
    hashes = {split: {sha256(p) for p in paths} for split, paths in source_paths.items()}
    if hashes["train"] & hashes["valid"]:
        raise ValueError("Image hash leakage between train and valid")
    if DATASET.exists():
        raise FileExistsError(f"Preserving existing dataset: {DATASET}")
    manifest, inputs, class_counts = [], [], Counter()
    for split, paths in source_paths.items():
        for kind in ("images", "labels"):
            (DATASET / split / kind).mkdir(parents=True)
        for source in paths:
            label_source = SOURCE / split / "labels" / (source.stem + ".txt")
            labels = read_labels(label_source)
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Cannot decode {source}")
            height, width = image.shape[:2]
            inputs.append(dict(split=split, image=str(source), label=str(label_source),
                               image_sha256=sha256(source), label_sha256=sha256(label_source)))
            crops = [("full", CropBox(0,0,width,height))]
            if split == "train":
                crops.extend((f"tile{i}", crop) for i, crop in enumerate(overlapping_tiles((width,height))))
            for tag, crop in crops:
                filename = f"{source.stem}__{tag}" + (source.suffix if tag == "full" else ".png")
                target = DATASET / split / "images" / filename
                target_label = DATASET / split / "labels" / (Path(filename).stem + ".txt")
                if tag == "full":
                    shutil.copy2(source, target)
                    shutil.copy2(label_source, target_label)
                    new_labels = labels
                else:
                    save_image(target, image[crop.y1:crop.y2,crop.x1:crop.x2])
                    new_labels = crop_labels(labels, (width,height), crop)
                    target_label.write_text("".join(f"{cls} {x:.9f} {y:.9f} {w:.9f} {h:.9f}\n" for cls,x,y,w,h in new_labels), encoding="utf-8")
                if split == "train":
                    class_counts.update(row[0] for row in new_labels)
                manifest.append(dict(split=split, source_image=source.name, generated_image=filename,
                                     kind=tag, **asdict(crop), labels=len(new_labels)))
    data = dict(path=str(DATASET.resolve()), train="train/images", val="valid/images", nc=len(names), names=names)
    (DATASET / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    write_csv(DATASET / "manifest.csv", manifest)
    dump(DATASET / "source_integrity.json", inputs)
    summary = dict(source_train_images=155, source_valid_images=20,
                   generated_train_images=sum(r["split"] == "train" for r in manifest),
                   generated_valid_images=sum(r["split"] == "valid" for r in manifest),
                   training_label_counts={names[c]: n for c,n in class_counts.items()},
                   tile_fraction=0.65, copies_are_independent=True, test_split_used=False)
    dump(DATASET / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def train(resume_imgsz=None, resume_reason="Resume from the last completed checkpoint"):
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this training configuration")
    settings = yaml.safe_load(TRAIN_CONFIG.read_text())
    run_dir = ROOT / settings["project"] / settings["name"]
    if run_dir.exists() and resume_imgsz is None:
        raise FileExistsError(f"Preserving existing run: {run_dir}")
    if not (DATASET / "summary.json").exists():
        raise RuntimeError("Prepare the crop dataset first")
    log_dir = ROOT / "outputs/experiments/crop-finetune-v1"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "training_status.json"
    previous_state = json.loads(state_path.read_text()) if state_path.exists() else None
    if resume_imgsz is not None:
        if not (run_dir/"weights/last.pt").is_file():
            raise FileNotFoundError("No checkpoint to resume")
        history_path = log_dir/"resume_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(dict(previous_status=previous_state, resumed_imgsz=resume_imgsz,
                            reason=resume_reason))
        dump(history_path,history)
        for name in ("best", "last"):
            backup=log_dir/f"before-resume-{len(history)}-{name}.pt"
            shutil.copy2(run_dir/f"weights/{name}.pt",backup)
    state = dict(status="starting", epochs=settings["epochs"], completed_epochs=0,
                 initialization=str(BASELINE), initialization_sha256=sha256(BASELINE),
                 expected_run_directory=str(run_dir), gpu=torch.cuda.get_device_name(0))
    state["training_imgsz"] = resume_imgsz or settings["imgsz"]
    state["resumed"] = resume_imgsz is not None
    dump(state_path, state)
    snapshot = {str(p.relative_to(ROOT)):p.read_text(encoding="utf-8") for p in (Path(__file__), ROOT/"src/billiards/crop_detection.py", TRAIN_CONFIG)}
    dump(log_dir / ("source_snapshot_resume.json" if resume_imgsz else "source_snapshot.json"), snapshot)
    console = sys.stdout
    start = time.perf_counter()

    def progress(trainer):
        state.update(status="training", completed_epochs=trainer.epoch+1,
                     elapsed_seconds=round(time.perf_counter()-start,1),
                     metrics={str(k):float(v) for k,v in trainer.metrics.items()})
        dump(state_path,state)
        print(json.dumps(state),file=console,flush=True)

    try:
        model = YOLO(str(run_dir/"weights/last.pt") if resume_imgsz else str(BASELINE))
        if resume_imgsz:
            state["completed_epochs"] = int(model.ckpt["epoch"]) + 1
            dump(state_path, state)
        model.add_callback("on_fit_epoch_end", progress)
        settings.pop("model", None)
        settings.pop("mode", None)
        settings.pop("task", None)
        settings["data"] = str((DATASET/"data.yaml").resolve())
        settings["project"] = str((ROOT/settings["project"]).resolve())
        with (log_dir / "training.log").open("a" if resume_imgsz else "w",encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            if resume_imgsz:
                model.train(resume=True,imgsz=resume_imgsz,workers=0,cache=False,plots=False,batch=1,device=0)
            else:
                model.train(**settings)
        best = run_dir/"weights/best.pt"
        if not best.is_file():
            raise RuntimeError("Training completed without a best checkpoint")
        state.update(status="complete", best_checkpoint=str(best), best_sha256=sha256(best),
                     elapsed_seconds=round(time.perf_counter()-start,1))
        if sha256(BASELINE) != state["initialization_sha256"]:
            raise RuntimeError("Baseline checkpoint was unexpectedly changed")
    except Exception as exc:
        state.update(status="failed", error=str(exc), elapsed_seconds=round(time.perf_counter()-start,1))
        dump(state_path,state)
        raise
    dump(state_path,state)
    print(json.dumps(state,indent=2),flush=True)


def convert(result, source):
    return [BoxDetection(p.class_id,p.box_xyxy,p.confidence,source) for p in result_to_predictions(result)]


def as_detection(point):
    return Detection(point.class_id, point.box_xyxy, point.confidence)


def evaluate(weights, output, reference):
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {output}")
    paths = sorted((SOURCE / "valid/images").glob("*.jpg"))
    if len(paths) != 20:
        raise RuntimeError("Expected all 20 validation images")
    output.mkdir(parents=True)
    (output / "overlays").mkdir()
    config = dict(weights=str(weights.resolve()), weights_sha256=sha256(weights), imgsz=960,
                  conf_min=.01, iou=.70, tile_fraction=.65, confidence_grid=CONFIDENCES,
                  main_full_confidence=.25, main_tile_confidence=.5, tolerances=[4.,8.,12.],
                  split="valid", test_split_used=False, reference=str(reference) if reference else None)
    dump(output/"config.json",config)
    reference_predictions = {}
    if reference:
        reference_predictions = {r["image"]:r for r in [json.loads(l) for l in (reference/"raw_predictions.jsonl").read_text().splitlines()]}
    raw_records, gt_by_image, gains, all_variants, tiles, timing_rows = [],[],[],[],[],[]
    model = YOLO(str(weights))
    dot_class = [i for i,n in model.names.items() if n == "Dot"]
    if len(dot_class)!=1:
        raise ValueError("A unique Dot class is required")
    dot_class = dot_class[0]
    for index,path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not decode {path}")
        height,width=image.shape[:2]
        start=time.perf_counter()
        full = convert(model.predict(image,imgsz=960,conf=.01,iou=.70,device=0,batch=1,verbose=False,save=False)[0],"full")
        full_time=time.perf_counter()-start
        tile_boxes=[]
        for i,crop in enumerate(overlapping_tiles((width,height))):
            tile=image[crop.y1:crop.y2,crop.x1:crop.x2]
            local=convert(model.predict(tile,imgsz=960,conf=.01,iou=.70,device=0,batch=1,verbose=False,save=False)[0],f"tile{i}")
            tile_boxes.extend(p.translate(crop.x1,crop.y1,f"tile{i}") for p in local)
        timing_rows.append(dict(image=path.name, full_seconds=full_time, total_seconds=time.perf_counter()-start))
        raw_records.append(dict(image=path.name,image_sha256=sha256(path),full=[asdict(p) for p in full],tiles=[asdict(p) for p in tile_boxes]))
        base_full=full
        if reference:
            ref=reference_predictions[path.name]
            if ref["image_sha256"]!=sha256(path):
                raise RuntimeError("Reference evaluation image mismatch")
            base_full=[BoxDetection(**p) for p in ref["full"]]
        variants={}
        for conf in CONFIDENCES:
            selected=[p for p in full if p.confidence>=conf]
            variants[f"full@{conf}"]=selected
            variants[f"tiles@{conf}"]=list(supplement_detections([], [p for p in tile_boxes if p.confidence>=conf]))
            variants[f"full+tiles@{conf}"]=list(supplement_detections(selected,[p for p in tile_boxes if p.confidence>=conf]))
        selected_base=[p for p in base_full if p.confidence>=.25]
        variants["baseline_full@0.25"]=selected_base
        variants["baseline_full0.25+tiles0.5"]=list(supplement_detections(selected_base,[p for p in tile_boxes if p.confidence>=.5]))
        variants["full0.25+tiles0.5"]=list(supplement_detections([p for p in full if p.confidence>=.25],[p for p in tile_boxes if p.confidence>=.5]))
        all_variants.append(variants)
        # Only scoring accesses validation labels.
        gt=load_ground_truth(path,width,height)
        gt_by_image.append(gt)
        gains.append(960/max(width,height))
        gt_dots=[p for p in gt if p.class_id==dot_class]
        panels=[draw_panel(image,gt_dots,[as_detection(p) for p in variants[name] if p.class_id==dot_class],gains[-1],8.,title)
                for name,title in (("baseline_full@0.25","Original full frame"),("full0.25+tiles0.5","Current full + crops"),("baseline_full0.25+tiles0.5","Original full + current crops"))]
        comparison=np.hstack(panels)
        save_image(output/"overlays"/path.name,comparison)
        tiles.append(cv2.resize(comparison,(960,round(comparison.shape[0]/2))))
        print(f"Inference {index+1}/20: {path.name}",flush=True)
    (output/"raw_predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in raw_records),encoding="utf-8")
    metrics=[]
    for name in all_variants[0]:
        for cls in range(len(model.names)):
            gt=[[p for p in rows if p.class_id==cls] for rows in gt_by_image]
            predictions=[[as_detection(p) for p in v[name] if p.class_id==cls] for v in all_variants]
            for tolerance in (4.,8.,12.):
                row=evaluate_setting(gt,predictions,gains,0.,tolerance)
                row.update(variant=name,class_name=model.names[cls])
                metrics.append(row)
    write_csv(output/"metrics.csv",metrics)
    write_csv(output/"timings.csv",timing_rows)
    dump(output/"summary.json",dict(config=config,image_count=20,metrics=metrics))
    save_image(output/"contact_sheet.jpg",np.vstack(tiles))
    selected=[r for r in metrics if r["class_name"]=="Dot" and r["tolerance_model_px"]==8]
    report=["# Aspect-preserving crop evaluation","",f"Checkpoint: {weights}",
            "All 20 validation images. Centre matching at 8px on 960 scale; no lattice or table-location filtering.","",
            "| Variant | TP | FP | FN | Precision | Recall | F1 | Complete images |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected:
        report.append(f"| {row['variant']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['fully_matched_images']}/20 |")
    report += ["", "Primary comparison was fixed before training: full-frame confidence 0.25 plus tile confidence 0.50.",
               "The confidence grid is a secondary diagnostic. Full-frame results are retained when crops overlap them.",
               "All five classes are evaluated by centre distance in metrics.csv; this is not a standard box-mAP or ball-projection evaluation.",
               "This is development-set evidence: validation images select training checkpoints and are not an independent test of generalization.",
               "Test and geometry holdout are unused. No geometry-quality claim is made. Crop inference requires four additional model passes."]
    (output/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(selected,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","train","resume","evaluate"))
    parser.add_argument("--resume-imgsz",type=int,default=640)
    parser.add_argument("--resume-reason",default="Resume from the last completed checkpoint")
    parser.add_argument("--weights",type=Path,default=BASELINE)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--reference",type=Path)
    args=parser.parse_args()
    if args.action=="prepare":
        prepare()
    elif args.action=="train":
        train()
    elif args.action=="resume":
        train(args.resume_imgsz, args.resume_reason)
    else:
        if args.output is None:
            parser.error("evaluate requires --output")
        evaluate(args.weights,args.output,args.reference)


if __name__=="__main__":
    main()
