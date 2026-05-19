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

# Run syndrome tests (requires OPENAI_API_KEY env var)
cd karyotype-analyzer-2 && python test_syndromes.py
cd karyotype-analyzer-2 && python test_triple_x_fix.py
```

## Architecture

### Single-File Application (`karyotype-analyzer-2/app.py`, ~5400 lines)

The entire application lives in one file. The major classes form a processing pipeline:

**Computer Vision Pipeline** (CV-based chromosome detection):
1. `DigitalPreprocessor` — Denoising (background subtraction → debris removal → CLAHE → bilateral smoothing) and chromosome straightening (skeleton extraction → perpendicular strip sampling)
2. `SegmentationMatrix` — Dual-path segmentation: semantic (pixel-level classification with stitching-based overlap recovery) and instance (marker-controlled watershed)
3. `ClusterRouter` — Classifies chromosome clusters via adjacency/BFS into isolated/touching/one-overlap/multi-overlap, routes each to the appropriate segmentation strategy
4. `ChromosomeClassifier` — 24-class classification (chr1-22, X, Y) using Gaussian similarity scoring on size (45%), centromere index (30%), banding (15%), aspect ratio (10%), with pair-based refinement enforcing autosome=2
5. `EnsembleClassifier` — 5-strategy ensemble (Simple CNN / Siamese Contrastive / SRAS-Enhanced / VariFocal Fusion / Multi-Task) with weighted majority voting
6. `ChromosomeDetector` — Orchestrates the full CV pipeline: denoise → threshold → route → segment → straighten → NMS → ensemble classify

**VLM Analysis Providers**:
- `KaryotypeAnalyzer` — Routes to provider-specific implementations (`_analyze_with_openai`, `_analyze_with_anthropic`, `_analyze_with_gemini`, `_mock_analysis`). Parses JSON from VLM responses using regex fallback (`r'\{[\s\S]*\}'`)
- `PrecisionClinicalLens` — 6-stage sequential VLM pipeline: counting → Denver group classification → individual chromosome ID → translocation detection → cross-validation → final ISCN notation. Each stage feeds results to the next

**API Provider Modes** (`APIProvider` enum):
- Single-model: OPENAI, ANTHROPIC, GEMINI
- Hybrid: CV_VLM (CV + VLM), TWO_STAGE (CV then VLM), CONSENSUS (multi-model voting)
- Advanced: PRECISION_LENS (6-stage pipeline)
- Demo: MOCK (no API needed)

**UI Layer** (Streamlit, lines ~4493-5430):
- Sidebar: provider selection, API key inputs, package status
- Main: image upload, analysis trigger, results display (ISCN notation, metrics, abnormalities)
- State: `st.session_state` for `analysis_result`, `uploaded_image`, `raw_response`

### API Integration

| Provider | Model | SDK Package |
|----------|-------|-------------|
| OpenAI | gpt-4o | `openai>=1.12.0` |
| Anthropic | claude-sonnet-4-20250514 | `anthropic>=0.18.0` |
| Google | gemini-2.0-flash | `google-genai>=1.0.0` |

All SDKs are optional — the app gracefully falls back with `try/except` imports and `*_AVAILABLE` flags.

## Key Concepts

- **ISCN 2020**: International System for Human Cytogenomic Nomenclature
- **Karyotype notation**: `{count},{sex_chromosomes}[,abnormalities]` (e.g., `47,XY,+21`)
- **Detectable abnormalities**: Trisomy (21, 18, 13), Monosomy X (Turner), translocations, deletions, inversions

## Development Notes

- Korean comments throughout codebase (페이지 설정 = page config, 디지털 전처리 = digital preprocessing)
- Image preprocessing: RGBA/P modes converted to RGB before base64 encoding
- Low temperature (0.1) for all VLM calls for deterministic analysis
- Deployment config exists for Railway (`Procfile`, `railway.toml`)


<!-- AUTOPUS:BEGIN -->
# Autopus-ADK Harness

> 이 섹션은 Autopus-ADK에 의해 자동 생성됩니다. 수동으로 편집하지 마세요.

- **프로젝트**: Karyogram
- **모드**: full
- **플랫폼**: claude-code, codex, gemini-cli

## 설치된 구성 요소

- Rules: .claude/rules/autopus/
- Skills: .claude/skills/autopus/
- Commands: .claude/skills/auto/SKILL.md
- Agents: .claude/agents/autopus/

## Language Policy

IMPORTANT: Follow these language settings strictly for all work in this project.

- **Code comments**: Write all code comments, docstrings, and inline documentation in English (en)
- **Commit messages**: Write all git commit messages in English (en)
- **AI responses**: Respond to the user in English (en)

## Core Guidelines

### Subagent Delegation

IMPORTANT: Use subagents for complex tasks that modify 3+ files, span multiple domains, or exceed 200 lines of new code. Define clear scope, provide full context, review output before integrating.

### File Size Limit

IMPORTANT: No source code file may exceed 300 lines. Target under 200 lines. Split source code by type, concern, or layer when approaching the limit. SPEC Markdown files under .autopus/specs/** are documentation and exempt from the 300-line source code limit. Excluded: generated files (*_generated.go, *.pb.go), documentation (*.md), and config files (*.yaml, *.json).

### Code Review

During review, verify:
- No source code file exceeds 300 lines (REQUIRED)
- SPEC Markdown files under .autopus/specs/** are not split or rejected for line count alone
- Complex changes use subagent delegation (SUGGESTED)
- See .claude/rules/autopus/ for detailed guidelines

<!-- AUTOPUS:END -->
