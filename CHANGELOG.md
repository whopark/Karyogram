# Changelog

## 2026-05-20

### SPEC-GRADCAM-001: CAM Visualization for Chromosome Classification Explainability

**New feature**: Class Activation Map visualization — heatmap overlays on chromosome crops showing which regions drove the classifier's prediction. Uses weight projection (Zhou et al. 2016), zero backward pass.

**Added**:
- `gradcam.py` (196 lines) — `generate_cam()` + `overlay_heatmap()` + CLI entry point (`python gradcam.py --model ... --image ... --target-class ...`)
- Streamlit UI integration: "Show CAM Heatmap" checkbox toggle in karyogram results expander
- Configurable target layer (`backbone.layer4` default, `backbone.layer3` for 8x8 resolution)
- 80th percentile threshold mask + inferno colormap (perceptually uniform, colorblind-friendly)

### SPEC-RESNET-001: Replace ChromosomeCNN with Pretrained ResNet18 Backbone

**Enhancement**: Replaced the custom 3-block CNN (ChromosomeCNN/ChromosomeNet) with a pretrained ResNet18 backbone for 24-class chromosome classification. Eliminates 3-way model duplication.

**Added**:
- `training/chromosome_model.py` — Single source of truth for model definitions (ChromosomeResNet18, legacy ChromosomeCNN, transforms, architecture detection)
- `training/__init__.py` — Package marker for cross-directory imports
- `karyogram_ui_models.py` — Extracted model loading from karyogram_ui.py
- Two-phase fine-tuning: frozen backbone warmup → full fine-tuning with differential LR
- ImageNet normalization (mean/std) applied to all input transforms
- `detect_architecture()` for backward-compatible weight loading (legacy CNN vs ResNet18)
- `torchvision>=0.17.0` and `torch>=2.2.0` dependencies
- `weights_only=True` on all `torch.load` calls (security hardening)

**Changed**:
- `training/train_classifier.py` — Imports from shared module, two-phase training
- `ml_pipeline.py` — Imports from shared module, auto-detect architecture on weight load
- `training/predict.py` — Imports from shared module, unified chrX/chrY label space
- `training/pair_trainer.py` — Added `warmup_epochs` parameter support
- `karyogram_ui.py` — Cache key bumped v3→v4, split into ui + ui_models

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

### SPEC-TRAIN-001: Paired Metaphase-Karyogram Training Pipeline

**New feature**: Supervised training from matched metaphase/karyogram image pairs. Parses clinical workstation screenshots to extract ground-truth labeled chromosome crops.

**Added**:
- `training/workstation_parser.py` — Splits clinical workstation screenshots (metaphase left + karyogram right), inverts dark backgrounds, extracts labeled chromosome crops
- `training/pair_trainer.py` — Orchestrates end-to-end paired training: filename matching with leading-zero normalization, crop extraction, YOLO annotation, dataset building with augmentation, sequential detector + classifier training, metrics reporting

**Results**: 10 matched pairs processed, ~20 crops per autosome class, YOLO mAP@0.5 improved from 0.376 to 0.606
