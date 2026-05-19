# SPEC-KARYO-001: Metaphase to Karyogram Generation UI

**Status**: completed
**Created**: 2026-05-19
**Domain**: KARYO

## Purpose

The existing Karyotype Analyzer application provides VLM-based and hybrid CV+VLM analysis modes that output ISCN notation from metaphase spread images, but none of them produce a **visual karyogram** -- the standard clinical representation where detected chromosomes are cropped, sorted by Denver group, and arranged in a labeled grid. This SPEC adds a complete end-to-end pipeline: YOLO-based detection, CNN-based 24-class classification, visual karyogram image generation, and Streamlit UI integration as a new analysis mode, so that a user uploading a metaphase spread image sees an arranged karyogram image alongside ISCN notation in the browser.

## Outcome Boundary

- **Mandatory outcome**: Upload a metaphase spread image in the Streamlit UI, select the "YOLO Karyogram" analysis mode, and receive a visual karyogram image (chromosomes arranged in standard 7-group rows with labels) plus ISCN 2020 notation displayed in the browser.
- **Completion evidence**: The karyogram image displays in the main content area with labeled chromosome pairs and Denver group separators; the ISCN notation, chromosome count, and sex determination are shown alongside the image.
- **Explicit non-goals**: Interactive chromosome re-assignment, ideogram band overlay, batch processing UI, side-by-side reference comparison, PDF/SVG export, replacing or modifying existing VLM-based analysis modes.

## Requirements

### REQ-01 — YOLO-based Chromosome Detection
Type: Event-Driven / Priority: Must

WHEN the user selects the "YOLO Karyogram" analysis mode and triggers analysis, THE SYSTEM SHALL load the YOLOv8 detector from the configured weights path and detect chromosome bounding boxes in the uploaded metaphase image with a configurable confidence threshold (default 0.25).

### REQ-02 — CNN-based 24-class Classification
Type: Event-Driven / Priority: Must

WHEN chromosome bounding boxes are detected by YOLO, THE SYSTEM SHALL crop each detection, resize to 64x32 grayscale, classify using the ChromosomeNet CNN model, and apply pair-based refinement to enforce autosome count of 2 per class.

### REQ-03 — Visual Karyogram Image Generation
Type: Event-Driven / Priority: Must

WHEN classified chromosome crops are available, THE SYSTEM SHALL arrange them into a standard karyogram grid image using PIL with 7 Denver group rows (A: chr1-3, B: chr4-5, C: chr6-12+X, D: chr13-15, E: chr16-18, F: chr19-20, G: chr21-22+Y), with each chromosome labeled by its class, homologous pairs placed side by side, and group separators drawn between Denver groups.

### REQ-04 — ISCN 2020 Notation Derivation
Type: Event-Driven / Priority: Must

WHEN chromosome classifications are finalized, THE SYSTEM SHALL derive total chromosome count, sex chromosome determination (XX/XY/other), and numeric abnormalities (trisomy, monosomy) to produce ISCN 2020 compliant notation in the format `{count},{sex}[,abnormalities]`.

### REQ-05 — Streamlit UI Integration as New Analysis Mode
Type: Event-Driven / Priority: Must

WHEN the application starts, THE SYSTEM SHALL register `YOLO_KARYOGRAM` as a new entry in the `APIProvider` enum, add it to the sidebar provider selection dropdown (visible when PyTorch and ultralytics packages are available), and route analysis to the ML pipeline when selected.

### REQ-06 — Karyogram Display and Download
Type: Event-Driven / Priority: Must

WHEN the karyogram image and ISCN notation are generated, THE SYSTEM SHALL display the karyogram image in the Streamlit main content area, show the ISCN notation, chromosome count, sex chromosomes, detected abnormalities, and Denver group distribution, and provide a download button for the karyogram PNG image.

### REQ-07 — Graceful Degradation on Missing Dependencies
Type: State-Driven / Priority: Must

WHILE PyTorch or ultralytics packages are not installed, THE SYSTEM SHALL hide the "YOLO Karyogram" option from the sidebar provider selection and preserve all existing analysis modes without error.

### REQ-08 — Graceful Degradation on Missing Weights
Type: State-Driven / Priority: Must

WHILE YOLO detector weights or CNN classifier weights are not found at the configured paths, THE SYSTEM SHALL display an informative error message in the Streamlit UI indicating which weight file is missing and how to obtain or train it, rather than crashing.

### REQ-09 — File Size Compliance
Type: Ubiquitous / Priority: Must

THE SYSTEM SHALL keep each new source code module under 300 lines.

## Generated File Summary

| File | Role | Lines (est.) |
|------|------|------------|
| `[NEW] karyotype-analyzer-2/ml_pipeline.py` | Orchestrates YOLO detection + CNN classification + pair refinement + ISCN derivation | ~200 |
| `[NEW] karyotype-analyzer-2/karyogram_generator.py` | Arranges classified chromosome crops into standard 7-row karyogram grid image using PIL | ~250 |
| `[NEW] karyotype-analyzer-2/karyogram_ui.py` | Streamlit UI components: progress display, karyogram image display, result summary, download | ~180 |
| `karyotype-analyzer-2/app.py` (modified) | Add `YOLO_KARYOGRAM` enum value, sidebar entry, and route to `karyogram_ui` | ~30 lines changed |

## Related SPECs

None.

## Traceability Matrix

| Requirement | Plan Task | Acceptance Scenario | Semantic Invariant |
|-------------|-----------|---------------------|--------------------|
| REQ-01 (YOLO detection) | T1, T2 | S1, S2 | INV-001, INV-002 |
| REQ-02 (CNN classification) | T1, T3 | S1, S3 | INV-003, INV-004 |
| REQ-03 (Karyogram image) | T4 | S1, S4 | INV-005, INV-006 |
| REQ-04 (ISCN notation) | T1 | S1, S5 | INV-007 |
| REQ-05 (UI integration) | T5, T6 | S1, S6 | -- |
| REQ-06 (Display and download) | T6 | S1, S7 | -- |
| REQ-07 (Dep degradation) | T5 | S8 | -- |
| REQ-08 (Weights degradation) | T2, T3 | S9 | -- |
| REQ-09 (File size) | T1-T6 | S10 | -- |
