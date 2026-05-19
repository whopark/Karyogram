# SPEC-TRAIN-001: Paired Metaphase-Karyogram Training Pipeline

**Status**: completed
**Created**: 2026-05-19
**Domain**: TRAIN

## Purpose

The existing training pipeline uses `karyogram_parser.py` to extract labeled chromosome crops from clean karyogram images (white background, isolated grid). The user's actual karyogram data consists of clinical workstation screenshots (~1800x950px RGBA) that contain a toolbar, dark background, a left half showing the metaphase spread, and a right half showing the arranged karyogram grid with numbered chromosome positions plus ISCN notation text at the bottom. There are also filename-matched metaphase images in a separate directory. The current parser cannot handle these workstation screenshots.

This SPEC adds a workstation screenshot parser and a paired training orchestrator so the user can run one command to extract labeled training data from matched pairs and train both the YOLO detector and the CNN classifier.

## Outcome Boundary

**User-visible outcome**: User runs `python pair_trainer.py --metaphase_dir ../metaphase --karyogram_dir ../karyogram` and the system automatically:
1. Finds filename-matched pairs (handling naming inconsistencies like `26-k-659` vs `26-k-0659`).
2. Splits each karyogram workstation screenshot into left (metaphase) and right (karyogram) halves.
3. Extracts labeled chromosome crops from the karyogram half using the existing `karyogram_parser.py`.
4. Generates YOLO annotations from the metaphase images using the existing `auto_annotate.py`.
5. Trains the YOLOv8 detector and the ChromosomeCNN classifier on the combined data.
6. Reports mAP and top-1/top-3 accuracy metrics to stdout.

**Mandatory requirements**: REQ-01 through REQ-08 below.

**Explicit non-goals**: Manual annotation UI, cloud/GPU training infrastructure, changes to the ChromosomeCNN architecture, changes to the YOLO model architecture, modification of existing working training scripts (`auto_annotate.py`, `karyogram_parser.py`, `build_dataset.py`, `train_detector.py`, `train_classifier.py`, `predict.py`).

**Completion evidence**: Pair trainer runs end-to-end on the 8 matched pairs, produces `chromosome_crops/` with per-class subdirectories, produces YOLO `annotations/`, trains both models, and prints metric summaries.

## Requirements

- **REQ-01** (Event-Driven / Priority: Must): WHEN the user provides a karyogram workstation screenshot, THE SYSTEM SHALL detect the vertical split boundary between the metaphase region (left) and the karyogram grid region (right) by analyzing column-wise pixel intensity variance, and crop the karyogram portion only.

- **REQ-02** (Event-Driven / Priority: Must): WHEN the cropped karyogram region has a dark background, THE SYSTEM SHALL invert the pixel intensities so chromosomes appear dark on a light background, then delegate to the existing `karyogram_parser.parse_karyogram()` for chromosome extraction.

- **REQ-03** (Event-Driven / Priority: Must): WHEN scanning paired directories, THE SYSTEM SHALL match filenames between `metaphase/` and `karyogram/` using a normalized stem comparison that strips leading zeros from numeric segments (e.g., `26-k-0659` matches `26-k-659`), and report unmatched files to stderr.

- **REQ-04** (Ubiquitous / Priority: Must): THE SYSTEM SHALL produce labeled chromosome crop images organized as `chromosome_crops/{chrN}/*.png` where N is 1-22, X, or Y, compatible with the existing `ChromosomeDataset` class in `train_classifier.py`.

- **REQ-05** (Event-Driven / Priority: Must): WHEN the user runs the pair trainer, THE SYSTEM SHALL generate YOLO bounding-box annotations for each metaphase image using the existing `auto_annotate.process_image()` function and save them under `annotations/labels/`.

- **REQ-06** (Event-Driven / Priority: Must): WHEN the labeled crops and YOLO annotations are ready, THE SYSTEM SHALL invoke `build_dataset.py` with `--augment --aug_factor 5` to build the YOLO dataset, then invoke `train_detector.py` and `train_classifier.py` sequentially.

- **REQ-07** (Event-Driven / Priority: Must): WHEN training completes, THE SYSTEM SHALL print a summary containing: number of matched pairs processed, total chromosome crops extracted per class, YOLO mAP@0.5 (or "N/A" if ultralytics is unavailable), CNN top-1 accuracy, and CNN top-3 accuracy.

- **REQ-08** (Event-Driven / Priority: Should): WHEN the extracted karyogram half contains fewer than 30 or more than 55 chromosome contours, THE SYSTEM SHALL log a warning with the filename and detected count but continue processing.

## Generated File Details

| File | Role | Lines (target) |
|------|------|----------------|
| `[NEW] training/workstation_parser.py` | Splits workstation screenshots, inverts dark backgrounds, delegates to `karyogram_parser.py` | <200 |
| `[NEW] training/pair_trainer.py` | Orchestrates paired training: scan dirs, match filenames, call workstation_parser, call auto_annotate, call build_dataset, call train_detector, call train_classifier, print metrics | <300 |

## Related SPECs

None

## Traceability Matrix

| Requirement | Plan Task | Acceptance Scenario | Semantic Invariant |
|-------------|-----------|---------------------|--------------------|
| REQ-01 | T1 | S1, S2 | INV-001 |
| REQ-02 | T1 | S2 | INV-002 |
| REQ-03 | T2 | S3, S4 | INV-003 |
| REQ-04 | T1, T3 | S5 | INV-004 |
| REQ-05 | T3 | S6 | - |
| REQ-06 | T4 | S7 | INV-005 |
| REQ-07 | T5 | S7, S8 | INV-006 |
| REQ-08 | T1 | S9 | - |
