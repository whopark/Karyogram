# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chromosome Karyotype Analyzer - A Streamlit web application for AI-powered cytogenetic analysis of metaphase spread images. Uses Vision-Language Models (GPT-4V, Claude, Gemini) to generate ISCN 2020 compliant karyotype notations.

## Commands

```bash
# Install dependencies
pip install -r karyotype-analyzer-2/requirements.txt

# Run the application
streamlit run karyotype-analyzer-2/app.py

# Run with specific port
streamlit run karyotype-analyzer-2/app.py --server.port 8501
```

## Architecture

### Single-File Structure (`karyotype-analyzer-2/app.py`)

**API Provider System**:
- `APIProvider` enum: OPENAI, ANTHROPIC, GEMINI, CONSENSUS, CV_VLM, TWO_STAGE, PRECISION_LENS, MOCK
- Graceful fallback with try/except imports for each SDK
- Package availability flags: `OPENAI_AVAILABLE`, `ANTHROPIC_AVAILABLE`, `GEMINI_AVAILABLE`

**SegmentationMatrix Class** (Task 2 - 얽힌 매듭 풀기):
- Path 1 - Semantic Segmentation: pixel-level background/chromosome/overlap classification, stitching-based overlap recovery using gradient direction
- Path 2 - Instance Segmentation: marker-controlled watershed with gradient-enhanced boundaries for per-instance masks
- `segment_and_separate()`: runs both paths and picks the best result

**ClusterRouter Class** (Task 3 - 사전 라우팅 메커니즘):
- Adjacency matrix construction via dilation overlap detection
- BFS-based connected component clustering
- 4-class classification: isolated / touching / one-overlap / multi-overlap
- Route A (touching) → instance segmentation, Route B (one-overlap) → semantic stitching, Route C (multi-overlap) → multi-pass

**DigitalPreprocessor Class** (Task 4 - 디지털 전처리):
- Cascaded Denoising: adaptive background subtraction → debris removal (inpainting) → CLAHE contrast → bilateral smoothing
- Chromosome Straightening: distance-transform skeleton → nearest-neighbor path ordering → perpendicular strip sampling
- Curvature measurement via discrete curvature formula on medial axis

**ChromosomeClassifier Class** (Task 5 - 24-클래스 분류):
- 24 reference templates (chr1-22, X, Y) with size_pct, centromere_index, Denver group
- Feature extraction: size ratio, centromere index (narrowest constriction), aspect ratio, super-resolved banding
- Gaussian similarity scoring with weighted combination (size 45%, centromere 30%, banding 15%, AR 10%)
- Pair-based refinement: enforces autosome=2 constraint, reassigns lowest-confidence over-represented classes
- Karyotype summary: ISCN notation, abnormality detection (trisomy/monosomy), sex determination

**EnsembleClassifier Class** (models.png - 5가지 분류 알고리즘 앙상블):
- Strategy 1: Simple CNN - size + centromere lightweight classification
- Strategy 2: Siamese Contrastive - cosine distance metric on banding embeddings
- Strategy 3: SRAS-Enhanced - super-resolution before feature extraction
- Strategy 4: VariFocal Fusion - global morphology + local banding band fusion
- Strategy 5: Multi-Task Ensemble - classification × segmentation quality × banding quality
- Weighted majority voting (15/20/20/25/20%) + pair-based refinement

**ChromosomeDetector Class** (Task 1-5 + models 통합):
- Pipeline: denoise → multi-threshold → ClusterRouter → SegmentationMatrix → overlap split → straighten+banding → 2-stage NMS → 5-strategy ensemble classify

**PrecisionClinicalLens Class** (6-stage sequential pipeline):
- Stage 1: 계수 (Counting) - Chromosome counting with optional CV pre-analysis
- Stage 2: 분류 (Classification) - Denver group classification (A-G)
- Stage 3: 클러스터 분류 (Cluster Classification) - Individual chromosome ID (1-22, X, Y)
- Stage 4: 전이 (Translocation Detection) - Structural abnormality detection
- Stage 5: 분석 (Comprehensive Analysis) - Cross-validation and synthesis
- Stage 6: 이상 탐지 (Abnormality Detection) - Final ISCN notation and clinical diagnosis
- Each stage builds on previous results; supports OpenAI, Anthropic, and Gemini backends

**KaryotypeAnalyzer Class**:
- Multi-provider analysis with `analyze()` method routing to provider-specific implementations
- `_analyze_with_openai()`: GPT-4o with high-detail image analysis
- `_analyze_with_anthropic()`: Claude Sonnet 4 via messages API
- `_analyze_with_gemini()`: Gemini 2.0 Flash via google-genai
- `_analyze_with_precision_lens()`: Routes to PrecisionClinicalLens 6-stage pipeline
- `_parse_response()`: Extracts JSON from raw model output with regex fallback
- `_mock_analysis()`: Demo mode with simulated karyotype results

**System Prompt** (lines 118-149):
- `KARYOTYPE_ANALYSIS_PROMPT`: Structured prompt requesting JSON output with ISCN notation, chromosome count, abnormalities, confidence score, and clinical interpretation

**Display Functions** (lines 416-784):
- `display_sidebar_settings()`: Provider selection dropdown, API key inputs with links
- `display_api_status()`: Shows installed/missing package status
- `display_results()`: ISCN notation, metrics, abnormalities, raw API response expander

**State Management**: `st.session_state` for `analysis_result`, `uploaded_image`, `raw_response`

## API Integration Details

| Provider | Model | SDK Package | Key Source |
|----------|-------|-------------|------------|
| OpenAI | gpt-4o | `openai>=1.12.0` | platform.openai.com |
| Anthropic | claude-sonnet-4-20250514 | `anthropic>=0.18.0` | console.anthropic.com |
| Google | gemini-1.5-pro | `google-generativeai>=0.4.0` | aistudio.google.com |

## Key Concepts

- **ISCN 2020**: International System for Human Cytogenomic Nomenclature
- **Karyotype notation**: `{count},{sex_chromosomes}[,abnormalities]` (e.g., `47,XY,+21`)
- **Detectable abnormalities**: Trisomy (21, 18, 13), Monosomy X (Turner), translocations, deletions, inversions

## Development Notes

- Korean comments throughout codebase (페이지 설정 = page config)
- Image preprocessing: RGBA/P modes converted to RGB before base64 encoding
- JSON extraction uses regex `r'\{[\s\S]*\}'` to handle markdown-wrapped responses
- Low temperature (0.1) for consistent, deterministic analysis results
