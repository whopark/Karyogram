# Changelog

## 2026-05-19

### SPEC-KARYO-001: Metaphase to Karyogram Generation UI

**New feature**: YOLO Karyogram analysis mode — upload a metaphase spread image and receive a visual karyogram (standard clinical 7-row grid layout) with ISCN 2020 notation.

**Added**:
- `ml_pipeline.py` — ML inference orchestrator (YOLO detection + CNN 24-class classification + pair refinement + ISCN derivation)
- `karyogram_generator.py` + `karyogram_render_helpers.py` — Visual karyogram renderer (PIL-based, standard Denver group layout with labels, separators, monosomy/trisomy markers)
- `karyogram_ui.py` — Streamlit UI components (progress display, karyogram image display, metrics, download)
- `APIProvider.YOLO_KARYOGRAM` enum + sidebar integration in `app.py`
- `training/` module — YOLO detector training, CNN classifier training, auto-annotation, dataset builder, karyogram parser, prediction pipeline

**Training pipeline**:
- `auto_annotate.py` — CV-based bounding box annotation from metaphase images
- `karyogram_parser.py` — Labeled chromosome crop extraction from karyogram images
- `build_dataset.py` — YOLO dataset assembly with augmentation
- `train_detector.py` — YOLOv8 fine-tuning for chromosome detection
- `train_classifier.py` — CNN training for 24-class chromosome classification
- `predict.py` — End-to-end inference pipeline (detect + classify + ISCN)
