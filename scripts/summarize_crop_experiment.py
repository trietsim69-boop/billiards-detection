"""Build matched-image before/after crop training and geometry diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from billiards.crop_detection import BoxDetection,supplement_detections
from billiards.geometry import PointObservation,fit_rails
from scripts.analyze_detector_errors import Detection,load_ground_truth
from scripts.analyze_dot_centers import write_csv
from scripts.sweep_rail_confidence import corner_alignment_error
from scripts.evaluate_rail_first import draw_panel,save_image,metric_row


def read_raw(directory):
    return {r["image"]:r for r in [json.loads(l) for l in (directory/"raw_predictions.jsonl").read_text().splitlines()]}


def detections(rows,confidence):
    return [BoxDetection(**r) for r in rows if r["confidence"]>=confidence and r["class_id"]==2]


def as_detection(p):
    return Detection(p.class_id,p.box_xyxy,p.confidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--training-status",type=Path,
                        default=ROOT/"outputs/experiments/crop-finetune-v1/training_status.json")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    before,after=read_raw(args.before),read_raw(args.after)
    before_config=json.loads((args.before/"config.json").read_text())
    after_config=json.loads((args.after/"config.json").read_text())
    for key in ("imgsz", "conf_min", "iou", "tile_fraction",
                "main_full_confidence", "main_tile_confidence", "tolerances", "split"):
        if before_config[key]!=after_config[key]:
            raise ValueError(f"Incomparable evaluation settings: {key}")
    training=json.loads(args.training_status.read_text())
    if set(before)!=set(after) or len(before)!=20:
        raise ValueError("Both runs must have the same complete validation set")
    args.output.mkdir(parents=True)
    (args.output/"overlays").mkdir()
    per_image,geometry,tiles=[],[],[]
    names=("original_full","original_full+original_crops","new_full","new_full+new_crops","original_full+new_crops")
    for filename,old in before.items():
        new=after[filename]
        if old["image_sha256"]!=new["image_sha256"]:
            raise ValueError("Source image changed between runs")
        path=ROOT/"data/processed/pix2pockets_v3/valid/images"/filename
        image=cv2.imread(str(path))
        height,width=image.shape[:2]
        gain=960/max(width,height)
        gt=[p for p in load_ground_truth(path,width,height) if p.class_id==2]
        gt_points=[PointObservation((p.box_xyxy[0]+p.box_xyxy[2])/2,(p.box_xyxy[1]+p.box_xyxy[3])/2) for p in gt]
        oracle=fit_rails(gt_points,(width,height))
        old_full,new_full=detections(old["full"],.25),detections(new["full"],.25)
        old_crops,new_crops=detections(old["tiles"],.5),detections(new["tiles"],.5)
        variants={names[0]:old_full,names[1]:supplement_detections(old_full,old_crops),
                  names[2]:new_full,names[3]:supplement_detections(new_full,new_crops),
                  names[4]:supplement_detections(old_full,new_crops)}
        panels=[]
        for name,points in variants.items():
            predictions=[as_detection(p) for p in points]
            metrics=metric_row(gt,predictions,gain,8.)
            fit=fit_rails([PointObservation(*p.center,p.confidence) for p in points],(width,height))
            error=corner_alignment_error(fit.corners,oracle.corners)
            per_image.append(dict(image=filename,variant=name,**metrics,
                                  geometry_valid=fit.valid,rails=len(fit.rails),
                                  mean_corner_error_px=error[0] if error else None,
                                  max_corner_error_px=error[1] if error else None))
            geometry.append(dict(image=filename,variant=name,fit=fit.as_dict(),
                                 dot_oracle=oracle.as_dict(),corner_error=error))
            if name in (names[0],names[1],names[3]):
                panels.append(draw_panel(image,gt,predictions,gain,8.,name))
        comparison=np.hstack(panels)
        save_image(args.output/"overlays"/filename,comparison)
        tiles.append(cv2.resize(comparison,(960,round(comparison.shape[0]/2))))
    summary=[]
    for name in names:
        rows=[r for r in per_image if r["variant"]==name]
        tp,fp,fn=(sum(r[key] for r in rows) for key in ("tp","fp","fn"))
        precision,recall=tp/max(1,tp+fp),tp/max(1,tp+fn)
        errors=[r["mean_corner_error_px"] for r in rows if r["geometry_valid"] and r["mean_corner_error_px"] is not None]
        summary.append(dict(variant=name,tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
                            f1=2*precision*recall/max(1e-12,precision+recall),
                            complete_images=sum(r["fn"]==0 for r in rows),
                            valid_geometry_images=sum(r["geometry_valid"] for r in rows),
                            median_corner_error_px=float(np.median(errors)) if errors else None))
    write_csv(args.output/"per_image.csv",per_image)
    write_csv(args.output/"summary.csv",summary)
    (args.output/"geometry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in geometry),encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (args.output/"provenance.json").write_text(json.dumps(dict(
        before=before_config,after=after_config,training_status=training),indent=2),encoding="utf-8")
    save_image(args.output/"contact_sheet.jpg",np.vstack(tiles))
    lines=["# Crop fine-tuning comparison", "",
           "All 20 validation images, 359 labelled dots. Full-frame confidence 0.25; supplementary crop confidence 0.50; centre tolerance 8px at 960 resolution. These settings were fixed before training.","",
           "| Method | TP | FP | FN | Precision | Recall | F1 | All dots found | Valid rail fits |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['variant']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['complete_images']}/20 | {r['valid_geometry_images']}/20 |")
    lines += ["", f"Evaluated fine-tuned checkpoint: `{after_config['weights']}` (SHA256 `{after_config['weights_sha256']}`).",
              "Training: original 960 checkpoint fine-tuned with 155 full training frames plus 620 labelled tiles. All five classes retained; AdamW initial learning rate 0.0003; 12-epoch budget and patience 5.",
              f"Training status: {training['status']}; completed epochs: {training['completed_epochs']}. Epoch 1 used 960 pixels. Following memory exhaustion, later epochs resumed at 640 pixels with optimizer state and reduced process-local thread counts. Every before/after inference comparison here uses 960 pixels.",
              "The trainer selects best.pt by standard all-class validation fitness, not the custom centre metric. Trainer scores mix 960 and 640 resolutions and cannot fairly rank checkpoints across that change; best.pt and last.pt are separately re-evaluated at 960. See the experiment RUN_NOTES.md for interruptions and checkpoint selection evidence.",
              "Geometry uses the unchanged fit_rails gates. A valid fit is structural evidence only, not certified bed geometry; corner error compares sight-line intersections to the Dot-label oracle.",
              "Crops use only image dimensions, not validation labels or table localization. Exact cross-split image hash overlap is rejected during preparation; original split membership is preserved.",
              "Validation is used for checkpoint selection, so this is development evidence. The detector test split and grouped geometry holdout remain unused.",
              "Overlays: original full frame, original model with crops, fine-tuned model with crops. Green=matched, red=missed, orange=false positive."]
    (args.output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
