# SPEC-KARYO-001 Research

## Existing Code Analysis

### Inference pipeline (training/predict.py)
- ChromosomeNet: 3-block CNN (Conv2d->BN->ReLU->Pool) with AdaptiveAvgPool2d(4,2) -> FC(1024->256->24). Input: 1-channel 64x32.
- IDX_TO_LABEL: 24 classes from chr1 to chr22, X, Y.
- _crop_tensor: crops bbox from RGB array, converts to grayscale, resizes to CROP_W=32 CROP_H=64, normalizes to 0-1, returns 1x1xHxW tensor.
- _refine_predictions: iterative reassignment of over-represented autosomes to under-represented ones (target=2 per autosome).
- _build_karyotype: derives sex (XX/XY/other), abnormalities, ISCN notation, Denver group counts.
- analyze_image: loads PIL image, runs YOLO detector, crops+classifies, refines, builds karyotype dict.

### Karyogram grid parser (training/karyogram_parser.py)
- Parses arranged karyogram images (the reverse operation of what we need to generate).
- KARYOGRAM_ORDER defines the standard row/group layout for 46/47 chromosomes.
- cluster_rows groups bounding boxes into horizontal rows by vertical center gap.
- Useful as a reference for the expected grid layout order.

### Existing app.py integration points
- APIProvider enum at line 46-54: 8 existing modes.
- display_sidebar_settings() at line 4532: builds provider_options list and provider_map dict.
- main() at line 5379: routes to display_analysis_section or specialized display functions.
- YOLO import guard already exists at lines 20-24 (YOLO_AVAILABLE flag).
- torch is NOT currently imported in app.py -- a new import guard is needed.

### Trained model weights (verified present)
- YOLO detector: karyotype-analyzer-2/runs/detect/training/runs/chromosome_v2/weights/best.pt (6.2 MB)
- CNN classifier: karyotype-analyzer-2/training/models/chromosome_classifier.pth (1.4 MB)

### Denver group constants
- A: chr1-3, B: chr4-5, C: chr6-12+X, D: chr13-15, E: chr16-18, F: chr19-20, G: chr21-22+Y

## Outcome Lock

- **User-visible outcome**: Upload metaphase spread image in Streamlit -> see arranged karyogram image with labeled chromosome pairs in standard Denver group rows + ISCN 2020 notation.
- **Mandatory requirements**: (1) YOLO-based chromosome detection, (2) CNN 24-class classification with pair refinement, (3) Visual karyogram image generation (standard 7-row grid layout), (4) Streamlit UI integration as new YOLO_KARYOGRAM analysis mode.
- **Explicit non-goals**: Interactive chromosome re-assignment, ideogram band overlay, batch processing UI, side-by-side reference comparison, PDF/SVG export, replacing or modifying any existing analysis mode.
- **Completion evidence**: User uploads metaphase PNG -> browser displays karyogram image with labeled chromosome pairs arranged in Denver group rows + ISCN notation + chromosome count + download button.

## Visual Planning Brief

The pipeline is a linear data flow from image upload to rendered karyogram:

```
Upload -> YOLO Detection -> Crop+Resize -> CNN Classification -> Pair Refinement -> ISCN Derivation -> Grid Layout -> Streamlit Display
```

Karyogram grid wireframe:

```
Row 1: [1a][1b] [2a][2b] [3a][3b] | [4a][4b] [5a][5b]     Groups A + B
Row 2: [6a][6b] [7a][7b] [8a][8b] [9a][9b]                 Group C part 1
Row 3: [10a][10b] [11a][11b] [12a][12b]                     Group C part 2
Row 4: [13a][13b] [14a][14b] [15a][15b]                     Group D
Row 5: [16a][16b] [17a][17b] [18a][18b]                     Group E
Row 6: [19a][19b] [20a][20b]                                Group F
Row 7: [21a][21b] [22a][22b] | [X][X] or [X][Y]            Group G + Sex
```

Each pair represents a homologous pair with the chromosome number label drawn below. Group separators are vertical dashed lines.

## Design Decisions

### Inlining vs importing from training/predict.py

The ChromosomeNet architecture, IDX_TO_LABEL, _refine_predictions, and _build_karyotype are inlined in ml_pipeline.py rather than imported from training/predict.py. Rationale:
- predict.py is a CLI module with argparse; importing it couples the web app to the training/ directory layout.
- The total inlined code is ~80 lines (model class + constants + 2 functions), well within the 300-line budget.
- This keeps the web-facing modules self-contained and deployable without the training directory.

### 3-module split vs single module

Three modules (ml_pipeline.py, karyogram_generator.py, karyogram_ui.py) rather than a single module because:
- Each module has a distinct responsibility (inference, rendering, UI).
- Each stays under 300 lines.
- The generator can be unit-tested independently with synthetic classification data.
- The UI module can be swapped for a different framework without touching inference or rendering.

### Minimal app.py changes

app.py is 5430 lines and not being refactored in this SPEC. Changes are limited to:
- 1 new import guard (~4 lines)
- 1 new enum value (~1 line)
- 2 new lines in provider_options/provider_map
- ~10 lines of routing logic in main()

This avoids regression risk in the existing monolith.

## Semantic Invariant Inventory

| ID | source clause | invariant type | affected outputs | acceptance IDs |
|----|---------------|----------------|------------------|----------------|
| INV-001 | YOLO-based chromosome detection | parser: YOLO bbox extraction from metaphase image | list of bounding boxes in detect_chromosomes return | S1, S2 |
| INV-002 | bounding boxes with width < 4 or height < 4 are filtered out | deduplication/filter: minimum bbox size constraint | filtered bbox list | S2 |
| INV-003 | classify using ChromosomeNet CNN model | paired matching: 24-class softmax + argmax classification | classification labels and confidence per crop | S1, S3 |
| INV-004 | pair-based refinement to enforce autosome count of 2 | grouping/constraint: autosome count invariant (exactly 2 per class) | refined label list | S3 |
| INV-005 | standard karyogram grid with 7 Denver group rows | ordering: chromosomes sorted by Denver group and number | karyogram image row layout | S1, S4 |
| INV-006 | homologous pairs placed side by side | paired matching: homolog pairing in visual grid | karyogram image pair positions | S4 |
| INV-007 | ISCN notation: count,sex,abnormalities | formula/format: ISCN 2020 notation derivation | notation string in result dict | S1, S5 |

## Feature Coverage Map

| Outcome slice | Covered by | Status |
|---------------|------------|--------|
| YOLO chromosome detection | Primary SPEC REQ-01, T1-T2 | covered |
| CNN 24-class classification + pair refinement | Primary SPEC REQ-02, T1-T3 | covered |
| Visual karyogram grid rendering | Primary SPEC REQ-03, T4 | covered |
| ISCN 2020 notation derivation | Primary SPEC REQ-04, T1 | covered |
| Streamlit UI integration (new mode) | Primary SPEC REQ-05-06, T5-T6 | covered |
| Graceful degradation (deps) | Primary SPEC REQ-07, T5 | covered |
| Graceful degradation (weights) | Primary SPEC REQ-08, T2-T3 | covered |
| File size compliance | Primary SPEC REQ-09, T1-T6 | covered |
| Happy path end-to-end | S1 | covered |
| Error/recovery paths | S8, S9 | covered |
| Download button | S7 | covered |

## Completion Debt

| Item | Blocks | Required resolution |
|------|--------|---------------------|
| None | - | - |

## Evolution Ideas

These are optional improvements and do not block sync completion.

| Idea | Why not required now | Promotion trigger |
|------|----------------------|-------------------|
| Interactive chromosome re-assignment UI | Outcome Lock explicitly excludes interactive editing | User explicitly requests drag-and-drop correction |
| Ideogram band overlay on karyogram | Visual enhancement only, not needed for arranged display | User requests band pattern reference overlay |
| Side-by-side reference karyogram comparison | Not part of the core generation pipeline | User requests comparison workflow |
| PDF/SVG vector export | PNG download satisfies the Outcome Lock | User needs print-quality or scalable output |
| Confidence heatmap overlay | Diagnostic tool, not part of visual karyogram display | User requests per-chromosome confidence visualization |

## Sibling SPEC Decision

| Decision | Reason | Sibling SPEC IDs |
|----------|--------|------------------|
| none | Primary SPEC closes Outcome Lock with 3 new modules + minimal app.py integration. Total ~630 new lines across 3 files, 6 tasks. | None |

## Reference Discipline

| Reference | Type | Verification |
|-----------|------|--------------|
| karyotype-analyzer-2/app.py::APIProvider line 46 | existing | verified via rg: 8 enum values at lines 46-54 |
| karyotype-analyzer-2/app.py::display_sidebar_settings line 4532 | existing | verified via rg: provider_options/provider_map at lines 4546-4567 |
| karyotype-analyzer-2/app.py::main line 5379 | existing | verified via rg: routing logic at lines 5397-5424 |
| karyotype-analyzer-2/app.py::YOLO_AVAILABLE line 22 | existing | verified via read: try/except block at lines 20-24 |
| karyotype-analyzer-2/training/predict.py::ChromosomeNet line 38 | existing | verified via read: CNN class at lines 38-56 |
| karyotype-analyzer-2/training/predict.py::IDX_TO_LABEL line 28 | existing | verified via read: 24-element list at line 28 |
| karyotype-analyzer-2/training/predict.py::_refine_predictions line 92 | existing | verified via read: iterative reassignment at lines 92-116 |
| karyotype-analyzer-2/training/predict.py::_build_karyotype line 123 | existing | verified via read: ISCN derivation at lines 123-156 |
| karyotype-analyzer-2/training/predict.py::CROP_H=64 CROP_W=32 line 31 | existing | verified via read: constants at line 31 |
| karyotype-analyzer-2/training/karyogram_parser.py::KARYOGRAM_ORDER line 21 | existing | verified via read: standard layout order at lines 21-31 |
| karyotype-analyzer-2/training/predict.py::DENVER_GROUPS line 21 | existing | verified via read: dict at lines 21-26 |
| runs/detect/training/runs/chromosome_v2/weights/best.pt | existing | verified via ls: 6.2 MB file present |
| training/models/chromosome_classifier.pth | existing | verified via ls: 1.4 MB file present |
| karyotype-analyzer-2/requirements.txt | existing | verified via read: streamlit, Pillow, opencv, numpy, openai, anthropic, google-genai |
| [NEW] karyotype-analyzer-2/ml_pipeline.py | planned addition | New file: ML inference orchestrator |
| [NEW] karyotype-analyzer-2/karyogram_generator.py | planned addition | New file: karyogram grid renderer |
| [NEW] karyotype-analyzer-2/karyogram_ui.py | planned addition | New file: Streamlit UI components |
| [NEW] karyotype-analyzer-2/app.py::APIProvider.YOLO_KARYOGRAM | planned addition | New enum value in existing class |
| [NEW] karyotype-analyzer-2/app.py::TORCH_AVAILABLE | planned addition | New import guard flag |

## Reviewer Brief

- **Intended scope**: Add YOLO+CNN karyogram generation as a new Streamlit analysis mode via 3 new modules (each under 300 lines) + minimal app.py wiring. Closes the Outcome Lock: upload metaphase -> see arranged karyogram image + ISCN notation.
- **Explicit non-goals**: No interactive editing, no ideogram overlay, no batch UI, no existing mode modifications, no app.py refactoring.
- **Self-verified**: Traceability Matrix (REQ-01..09 -> T1..6 -> S1..10 -> INV-001..007), Semantic Invariant Inventory (7 invariants with oracle acceptance), existing/[NEW] Reference Discipline (14 existing verified + 5 planned additions), all oracle acceptance scenarios have concrete expected values.
- **Reviewer focus**: Correctness of ChromosomeNet inlining (must match training/predict.py architecture exactly), karyogram grid row composition matches Denver group standard, backward compatibility of app.py changes (no existing mode breakage), file size compliance.

## Technology Stack Decision

| Mode | Selected stack | Resolved versions | Source refs | Checked at | Rejected alternatives |
|------|----------------|-------------------|-------------|------------|-----------------------|
| brownfield | Python | 3.13 existing project | Existing project runtime | 2026-05-19 | N/A existing project |
| brownfield | Streamlit | >=1.31.0 existing constraint | requirements.txt line 2 | 2026-05-19 | N/A existing dependency |
| brownfield | PyTorch | 2.7.1 existing training env | Trained weights compatible | 2026-05-19 | N/A weights already trained |
| brownfield | ultralytics YOLOv8 | existing yolov8n.pt | chromosome_v2/args.yaml | 2026-05-19 | N/A detector already trained |
| brownfield | Pillow | >=10.2.0 existing | requirements.txt line 5 | 2026-05-19 | N/A existing dependency |
| brownfield | OpenCV | >=4.8.0 existing | requirements.txt line 6 | 2026-05-19 | N/A existing dependency |

All dependencies are brownfield. No new packages are introduced. PyTorch and ultralytics are optional runtime dependencies (graceful degradation when absent).

## Self-Verify Summary

- Q-CORR-01 | status: PASS | attempt: 1 | files: research.md | reason: All existing references verified via rg/read/ls against actual codebase paths, line numbers, and symbols.
- Q-CORR-02 | status: PASS | attempt: 1 | files: spec.md, plan.md, research.md | reason: All 5 planned additions marked with [NEW] prefix; no planned file treated as existing.
- Q-CORR-03 | status: PASS | attempt: 1 | files: spec.md, acceptance.md | reason: EARS requirements use WHEN/THE SYSTEM SHALL form; acceptance uses bare Given/When/Then; Priority uses Must/Should/Nice only.
- Q-CORR-04 | status: PASS | attempt: 1 | files: research.md | reason: Reference Discipline separates 14 existing references from 5 [NEW] planned additions.
- Q-COMP-01 | status: PASS | attempt: 1 | files: all | reason: All four SPEC files present with distinct roles.
- Q-COMP-02 | status: PASS | attempt: 1 | files: spec.md, acceptance.md | reason: Traceability Matrix maps all 9 requirements to plan tasks and acceptance scenarios.
- Q-COMP-03 | status: PASS | attempt: 1 | files: spec.md | reason: Each requirement states EARS type, trigger condition, expected behavior, and observable outcome.
- Q-COMP-04 | status: PASS | attempt: 1 | files: research.md, spec.md, acceptance.md | reason: Outcome Lock defines mandatory outcome, non-goals, and completion evidence. REQ-01..06 and S1 close the mandatory requirements.
- Q-COMP-05 | status: PASS | attempt: 1 | files: research.md, acceptance.md | reason: All 7 semantic invariants map to requirements, plan tasks, and acceptance scenarios with oracle acceptance containing concrete expected values.
- Q-COMP-06 | status: PASS | attempt: 1 | files: spec.md, research.md | reason: Traceability Matrix links REQ->Task->Scenario->Invariant. Reviewer Brief states scope, non-goals, self-verified evidence, and reviewer focus.
- Q-COMP-07 | status: PASS | attempt: 1 | files: research.md | reason: Completion Debt is empty. Evolution Ideas contains 5 optional improvements without SPEC IDs, task IDs, or acceptance IDs.
- Q-FEAS-01 | status: PASS | attempt: 1 | files: plan.md | reason: All changes are runtime code. No documentation-only claims for runtime behavior.
- Q-FEAS-02 | status: PASS | attempt: 1 | files: plan.md, research.md | reason: All new files created in karyotype-analyzer-2/ module root. app.py changes target the existing file.
- Q-FEAS-03 | status: PASS | attempt: 1 | files: acceptance.md | reason: All verification steps runnable: line count check, Streamlit manual testing, import checks, file existence checks.
- Q-STYLE-01 | status: PASS | attempt: 1 | files: spec.md | reason: All requirement descriptions use definitive language (SHALL). No ambiguous should/might/could.
- Q-STYLE-02 | status: PASS | attempt: 1 | files: spec.md | reason: Priority (Must) and EARS type are on separate annotation lines.
- Q-STYLE-03 | status: PASS | attempt: 1 | files: acceptance.md | reason: All scenarios use bare Given/When/Then/And format without bullet or bold markup.
- Q-SEC-01 | status: PASS | attempt: 1 | files: spec.md, research.md | reason: Trust boundary addressed: user-uploaded images are external input. No prompt injection surface. Existing 10MB limit applies.
- Q-SEC-02 | status: PASS | attempt: 1 | files: plan.md, research.md | reason: Weight paths relative to project, not user-controlled. No API keys or credentials in YOLO_KARYOGRAM mode.
- Q-SEC-03 | status: N/A | attempt: 1 | files: all | reason: No logging artifacts or persistent files created. In-memory karyogram image only. Download is user-initiated.
- Q-COH-01 | status: PASS | attempt: 1 | files: all | reason: Single cohesive change story: metaphase-to-karyogram pipeline. All 3 new modules serve the same feature.
- Q-COH-02 | status: PASS | attempt: 1 | files: research.md | reason: No follow-on runtime work needed. Trained weights exist. All mandatory requirements covered. Evolution Ideas are advisory only.
- Q-COH-03 | status: PASS | attempt: 1 | files: research.md | reason: No sibling SPECs. Primary SPEC closes Outcome Lock with 6 tasks, 3 new files.
