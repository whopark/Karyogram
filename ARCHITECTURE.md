# Architecture — Karyogram

## Overview

Chromosome Karyotype Analyzer: a Streamlit web application for AI-powered cytogenetic analysis of metaphase spread images. Uses Vision-Language Models (GPT-4V, Claude, Gemini) and computer vision to generate ISCN 2020 compliant karyotype notations.

## Domain Map

```
┌─────────────────────────────────────────────────────────┐
│                    Karyogram                            │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  CV Pipeline  │  │ VLM Analysis │  │  UI Layer    │  │
│  │              │  │              │  │ (Streamlit)  │  │
│  │ Preprocessor │  │ OpenAI       │  │              │  │
│  │ Segmentation │  │ Anthropic    │  │ Sidebar      │  │
│  │ ClusterRouter│  │ Gemini       │  │ Upload       │  │
│  │ Classifier   │  │ Consensus    │  │ Results      │  │
│  │ Ensemble     │  │ PrecisionLens│  │ Report       │  │
│  │ Detector     │  │              │  │ KaryogramUI  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│         └────────┬────────┘                  │          │
│                  ▼                           │          │
│         ┌──────────────┐                     │          │
│         │KaryotypeAnalyzer│◄─────────────────┘          │
│         │  (Router)    │                                │
│         └──────────────┘                                │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ML Pipeline (YOLO Karyogram Mode)               │   │
│  │  ml_pipeline.py → karyogram_generator.py         │   │
│  │  YOLO detect → CNN classify → karyogram render   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Layers

| Layer | Location | Responsibility |
|-------|----------|---------------|
| **CV Pipeline** | `app.py:57-3125` | Image preprocessing, chromosome segmentation, classification |
| **VLM Analysis** | `app.py:2970-4492` | Multi-provider AI vision analysis with ISCN notation generation |
| **UI** | `app.py:4493-5430` | Streamlit interface: upload, settings, results display |

## Key Classes

| Class | Lines | Role |
|-------|-------|------|
| `DigitalPreprocessor` | 57-378 | 4-stage cascaded denoising + chromosome straightening |
| `SegmentationMatrix` | 379-767 | Dual-path segmentation (semantic + watershed) |
| `ClusterRouter` | 768-1156 | Routes clusters by overlap type to segmentation strategy |
| `ChromosomeClassifier` | 1157-1596 | 24-class Gaussian similarity scoring with ISCN templates |
| `EnsembleClassifier` | 1597-1993 | 5-strategy weighted majority voting ensemble |
| `ChromosomeDetector` | 1994-3125 | Orchestrates full CV pipeline |
| `PrecisionClinicalLens` | 3126-3555 | 6-stage sequential VLM pipeline |
| `KaryotypeAnalyzer` | 3556-4492 | Top-level router to provider-specific analysis methods |

## API Provider Modes

| Mode | Enum | Description |
|------|------|-------------|
| Single-model | OPENAI, ANTHROPIC, GEMINI | Direct VLM analysis |
| Hybrid | CV_VLM, TWO_STAGE | CV detection + VLM interpretation |
| Multi-model | CONSENSUS | Multi-provider voting |
| Advanced | PRECISION_LENS | 6-stage sequential VLM pipeline |
| ML Pipeline | YOLO_KARYOGRAM | YOLO detection + ResNet18/CNN classification + visual karyogram |
| Demo | MOCK | No API needed |

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit>=1.31.0` | Web framework |
| `openai>=1.12.0` | GPT-4o vision |
| `anthropic>=0.18.0` | Claude vision |
| `google-genai>=1.0.0` | Gemini vision |
| `opencv-python-headless>=4.8.0` | Computer vision |
| `Pillow>=10.2.0` | Image processing |
| `numpy>=1.24.0` | Numerical compute |
| `torch>=2.2.0` | ML inference (optional, for YOLO Karyogram mode) |
| `torchvision>=0.17.0` | Pretrained ResNet18 backbone (optional, for CNN classifier) |
| `ultralytics>=8.0.0` | YOLOv8 object detection (optional) |

All VLM SDKs and ML packages are optional with `try/except` import guards and `*_AVAILABLE` flags.

## ML Pipeline Modules (SPEC-KARYO-001, SPEC-RESNET-001)

| Module | Lines | Role |
|--------|-------|------|
| `training/chromosome_model.py` | 141 | Shared model definitions: ChromosomeResNet18, legacy ChromosomeCNN, transforms, architecture detection |
| `ml_pipeline.py` | 281 | YOLO detection + CNN/ResNet18 classification + ISCN derivation (backward-compat weight loading) |
| `karyogram_generator.py` | 141 | Karyogram layout logic + public API |
| `karyogram_render_helpers.py` | 221 | PIL rendering primitives (rows, pairs, grid) |
| `karyogram_ui.py` | 293 | Streamlit UI for karyogram mode |
| `karyogram_ui_models.py` | 43 | Model loading + progress display (extracted from karyogram_ui) |

## Deployment

- **Platform**: Railway (nixpacks builder)
- **Start**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Health check**: `GET /` with 300s timeout

## Structural Debt

- `app.py` is 5,430 lines (single-file monolith) — pre-existing, acknowledged in CLAUDE.md
- No structured test framework (pytest); only ad-hoc test scripts
- No lock file for reproducible builds (only `requirements.txt`)
