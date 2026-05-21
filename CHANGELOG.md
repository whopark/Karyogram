# Changelog

## 2026-05-21

### SPEC-ENHANCE-001: FLUX.1 Image-to-Image Karyogram Enhancement

**New feature**: Optional post-processing step that sends the raw karyogram to FLUX.1 dev img2img API (via fal.ai) for a visually polished, clinically presentable version.

**Added**:
- `karyogram_enhance.py` (108 lines) — FLUX.1 API integration: CDN upload, img2img call, image download, graceful fallback
- "Enhance Karyogram (FLUX.1)" checkbox toggle in karyogram UI
- `fal-client>=1.0.0` as optional dependency

**Changed**:
- `karyogram_ui.py` — added `_try_enhance()` helper for conditional FLUX.1 enhancement after `generate_karyogram()`
- `karyogram_ui_models.py` — moved `display_karyogram_download()` from karyogram_ui.py for line budget management
- All files remain under 300-line limit

### SPEC-SEX-001: Sex-Aware Pair Refinement for chrX/chrY Recognition

**Enhancement**: Added sex chromosome assignment logic to `pair_refine` post-processing, using elimination + crop size heuristic.

**Added**:
- `ml_refine.py` (79 lines) — extracted `pair_refine` function with two-phase refinement
  - Phase 1: autosome pairing (each=2, existing logic)
  - Phase 2: sex chromosome assignment by elimination from Group C/G + crop area size discrimination
- `classify_chromosomes` now computes crop areas from YOLO bbox dimensions and passes to `pair_refine`
- Backward-compatible `crop_areas=None` default parameter

**Changed**:
- `ml_pipeline.py` reduced from 337 to 257 lines (pair_refine extracted to ml_refine.py)
- `ml_pipeline.py` imports `pair_refine` from `ml_refine` instead of inline `_pair_refine`

### SPEC-UX-001: Collapsible Sidebar Sections

**Enhancement**: Sidebar reorganized with `st.expander` for progressive disclosure.

**Changed**:
- Provider selector always visible at top (outside expanders)
- API key input wrapped in collapsed "API Key" expander
- Package status and About section wrapped in collapsed "Status & Info" expander
- Removed `st.divider()` calls (expanders provide visual separation)
- `display_api_status()` moved inside Status & Info expander

## 2026-05-20

### SPEC-SPLIT-001: Split app.py Monolith into Multi-Module Architecture

**Refactor**: Decomposed the 5,452-line single-file monolith into domain-organized packages.

**Added**:
- `providers.py` (57 lines) — centralized APIProvider enum and import guards
- `cv/` package (14 files, 2,303 lines) — DigitalPreprocessor, SegmentationMatrix, ClusterRouter, ChromosomeClassifier, EnsembleClassifier, ChromosomeDetector with helper splits
- `vlm/` package (6 files, 1,120 lines) — KaryotypeAnalyzer, PrecisionClinicalLens, prompts with helper splits
- `ui/` package (10 files, 799 lines) — sidebar, upload, analysis, results, report, styles with helpers

**Changed**:
- `app.py` rewritten as 101-line thin entry point (from 5,452 lines, -98%)
- Updated external script imports: `run_all_analyses.py`, `run_pipeline.py`, `debug_cv_detection.py`, `test_triple_x_fix.py`
- All new files under 300-line hard limit (max: 283 lines in vlm/analyzer.py)
- Zero behavioral changes — E2E scenarios 10/10 pass

### SPEC-TRAIN-002: Enhanced Training Recipe for ChromosomeResNet18

**Enhancement**: Improved classifier training with label smoothing, focal loss, enhanced augmentation, and optional Mixup.

**Added**:
- `FocalLoss` class in `chromosome_model.py` with configurable gamma, alpha, and label smoothing
- `mixup_data()` helper for Mixup augmentation (Beta(0.2, 0.2))
- `--loss-type {ce,focal}` CLI flag for loss function selection
- `--mixup` CLI flag for optional Mixup augmentation in both training phases
- Enhanced augmentation: RandomRotation(30), RandomAffine, RandomErasing(0.3)
- Label smoothing (0.1) applied to default CrossEntropyLoss

**Changed**:
- Refactored `compute_class_weights` from `train_classifier.py` to `chromosome_model.py`
- Phase 2 now reloads best warmup checkpoint before fine-tuning
- Fixed `FocalLoss` reduction="none" to return per-element tensor

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
