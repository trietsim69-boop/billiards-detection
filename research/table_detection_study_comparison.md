# Table-detection study comparison

_Prepared 2026-08-24. Scope: primary sources that report billiards-ball detection, rail-dot/table localization, or closely related tracking results._

## Short answer

Yes: the current dot recall is low relative to the only close published comparison I found. The Pix2Pockets paper reports **91.6% raw dot recall** and **90.8% after post-processing**, while our current YOLO11n validation result is **64.0%** (or **65.7%** at the confidence-0.50 operating point used by the error-analysis script).

That is a gap of **27.6 percentage points** against the paper's raw result. It is meaningful, but it is not yet a controlled head-to-head result: the paper reports on its **test set**, whereas ours is a **validation-set** result; the exact image membership and evaluation confidence are not published; and the models and training schedules differ.

## Closest comparison: Pix2Pockets

The peer-reviewed SCIA 2025 paper, [Pix2Pockets: Shot Suggestions in 8-Ball Pool](https://link.springer.com/chapter/10.1007/978-3-031-95911-0_29), is the source of this project's dataset and is therefore the most relevant benchmark. The full primary manuscript is available on [arXiv](https://arxiv.org/html/2504.12045), with code in the [authors' repository](https://github.com/viktorseba/pix2pockets).

### Dot comparison

| Metric | Current YOLO11n, validation | Pix2Pockets YOLOv5, test (raw) | Pix2Pockets YOLOv5, test (post-processed) |
|---|---:|---:|---:|
| Precision | 83.3% | 83.5% | 90.8% |
| Recall | **64.0%** | **91.6%** | **90.8%** |
| AP@0.50 | 64.0% | 89.5% | 89.3% |
| AP@0.50:0.95 | 30.8% | 51.0% | 51.4% |

Our separate confidence-0.50 error analysis counted 236 TP, 51 FP, and 123 FN for dots: precision 82.2%, recall 65.7%, and F1 73.1%. This operating-point result should not be substituted for Ultralytics' confidence-swept AP or for the paper's result without matching its threshold.

### All Pix2Pockets detection results

Each paper cell below is shown as **raw → after post-processing**. Its post-processing applies class-agnostic NMS and billiards-specific caps on physically possible detections.

| Class | Precision | Recall | F1 | AP@0.50 | AP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|
| Striped balls | 84.0 → 97.8 | 89.9 → 89.9 | 86.8 → 93.7 | 90.0 → 91.4 | 79.1 → 80.4 |
| Solid balls | 90.0 → 95.3 | 89.0 → 89.0 | 89.5 → 92.0 | 89.9 → 90.4 | 80.7 → 80.9 |
| Cue ball | 86.4 → 95.0 | 95.0 → 95.0 | 90.5 → 95.0 | 92.2 → 92.3 | 83.6 → 84.0 |
| Black ball | 85.7 → 100.0 | 94.7 → 94.7 | 90.0 → 97.3 | 92.2 → 92.4 | 84.9 → 85.4 |
| Rail dots | 83.5 → 90.8 | 91.6 → 90.8 | 87.4 → 90.8 | 89.5 → 89.3 | 51.0 → 51.4 |
| Mean | 85.9 → 95.8 | 92.0 → 91.9 | 88.8 → 93.8 | 90.8 → 91.2 | 75.9 → 76.4 |

The low dot AP@0.50:0.95 relative to the ball classes is important: even the published model localizes tiny dots less precisely at stricter IoU thresholds. However, its much higher recall shows that our present model is also missing substantially more dots, not merely drawing less precise boxes.

### Evaluation and training setup

- Dataset: 195 broadcast/in-the-wild images, 5,748 annotations, and the same five classes used here.
- Split sizes: 155 train, 20 validation, and 20 test images.
- Paper model: pretrained YOLOv5, input resized to 640 × 640, batch size 20, learning rate 0.01, and 2,000 training epochs. The exact YOLOv5 size/variant is not stated.
- Current model: YOLO11n at 640 × 640, batch size 2, maximum 100 epochs; early stopping ended training after 66 epochs, with best epoch 46.
- Current metrics above are from our 20-image validation split. The Pix2Pockets metrics are from its 20-image test split. The paper does not publish the exact files in each split or the confidence threshold used for its precision/recall table.
- Roboflow hosts the [8-Ball Pool dataset](https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o), but its public page does not provide a separate official trained-model benchmark to compare with the paper.

### Geometry result from Pix2Pockets

For a separate 52-image geometry dataset containing 25 matched table situations, Pix2Pockets fits four rail lines with RANSAC from detected dots, infers corners, estimates a homography from up to 22 dot correspondences, and reports **0.4 cm mean projected ball-position error** on a normalized nine-foot table (0.22 cm in its best illustrated example and 0.76 cm in its worst).

This is an end-to-end projection-error result, **not** rail-dot recall, corner error, or an automatic homography success rate. The paper does not report those latter metrics.

## Other primary billiards studies

These studies help establish what has been attempted, but their numbers should not be used as direct targets for our five-class broadcast-image detector.

| Study | Setup and task | Exact reported result | Comparable detector recall? |
|---|---|---|---|
| Gao et al., [A Novel Computer Vision-Based Automatic Detection System for Billiards](https://journalskuwait.org/kjs/index.php/KJS/article/view/2083) (Kuwait Journal of Science, 2018) | Controlled overhead CCD; normalized-RGB segmentation, improved Hough transform, and least-squares positioning; ball-pattern classification is a separate stage. | 99.4% “detection accuracy,” average 0.65 s; solid/striped recognition above 98.5%. | **No.** The paper does not report precision, recall, IoU, AP, rail-dot detection, or homography accuracy. |
| Pan et al., [Intelligent recognition and positioning of billiards based on machine vision](https://www.nature.com/articles/s41598-024-63955-3) (Scientific Reports, 2024) | Controlled overhead CCD and a two-thirds-size table; classical segmentation/positioning followed by an Xception-based solid/striped crop classifier. The classifier test set contains 100 12 × 12 ball crops after training on 400 crops. | Method-comparison table reports a 97.81% ball-detection rate and about 0.5 s; final test recognition is 98.65% for solid balls and 98.927% for striped balls. | **No standard recall.** The figures mix segmentation success and cropped-ball classification and do not cover dots. |
| Sousa et al., [A robust vision system for intelligent cue sports](https://link.springer.com/article/10.1007/s41095-016-0047-3) (Computational Visual Media, 2016) | Fixed overhead Kinect 2 plus projector; RGB-D table/ball localization and shot projection. Evaluation includes 163 shots and 605 stated ball-detection opportunities. | Table reports ball detection 97.2% (588/605), cue-ball classification 91.0% (354/389), table-boundary success 100% in three trials with <3 px error, and maximum ball-position error 10 mm. | **Not standard recall/AP.** The prose also says all 605 balls were detected plus 17 false positives, which conflicts with interpreting 588/605 as ordinary recall. No rail dots. |
| Rodriguez-Lozano et al., [MOLT: a multi-objective system for billiards tracking](https://link.springer.com/article/10.1007/s10489-023-04542-3) (Applied Intelligence, 2023) | Fixed overhead 640 × 480 video at 20 fps; 300 shots and >85,000 frames across blackball, carom, and snooker; evaluates tracking trajectories. | Tracking Jaccard is approximately 0.86 and above 0.85 across modalities; users scored the 3-D reconstruction 7.6/10 overall. | **No.** These are video tracking/reconstruction metrics, not single-image detector recall or rail-dot localization. |
| Legg et al., [Accurate object localization in 3D using a single camera](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2011/papers/1569407575.pdf) (ICIP, 2011) | Tripod-view snooker at 30 fps; perspective correction, HSL/specular-highlight ball detection, classification, and tracking. | Qualitative examples and failure discussion; no precision, recall, AP, or per-class detection table. | **Unavailable.** |
| Silva et al., [A system for recommending strategic shots in billiards](https://ebuah.uah.es/xmlui/bitstream/handle/10017/64174/System_Silva_ICOA_2024.pdf?isAllowed=y&sequence=3) (2024) | Overhead GoPro and YOLOv8n for cue/balls, with manually selected calibration points; evaluates a downstream shot-recommendation system. | No detector dataset size, precision, recall, mAP, or per-class results are reported. | **Unavailable.** Its participant-level recommendation outcomes are not detector metrics. |

## Interpretation for this project

1. **Treat dot recall as the current detector bottleneck.** Ball recalls are roughly 83–86% in our validation results, while dots are 64.0%; Pix2Pockets obtains about 91% dot recall on its test data.
2. **Do not benchmark against the 97–99% figures from controlled-camera papers.** Those studies use overhead cameras, controlled lighting/table geometry, classical segmentation, cropped-ball classification, or tracking/system success measures. None reports five-class rail-dot recall on broadcast images.
3. **Use Pix2Pockets as a directional reference, not a pass/fail threshold yet.** A fair reproduction requires either recovering its split and evaluation settings or evaluating both models on the same untouched images with identical matching and confidence rules.
4. **Measure geometry-oriented dot success in addition to box recall.** Before retraining, record center-distance accuracy and whether each rail has enough correctly ordered dots for line fitting. A small or low-IoU dot box can still be adequate for homography if its center is accurate; a missed rail or clustered false positives can be fatal even when aggregate recall looks acceptable.

