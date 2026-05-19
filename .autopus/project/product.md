# Product — Karyogram

## Name
Chromosome Karyotype Analyzer (Karyogram)

## Description
AI-powered web application for cytogenetic analysis of metaphase spread images. Analyzes chromosome karyotype images using Vision-Language Models and computer vision to generate ISCN 2020 compliant karyotype notations.

## Core Features

1. **Multi-Provider VLM Analysis**: Supports OpenAI (GPT-4o), Anthropic (Claude), and Google (Gemini) for karyotype analysis
2. **Computer Vision Pipeline**: Automated chromosome detection, segmentation, and classification using OpenCV
3. **Hybrid Analysis Modes**: CV+VLM, Two-Stage, Consensus (multi-model voting), Precision Clinical Lens (6-stage pipeline)
4. **ISCN 2020 Notation**: Generates standardized cytogenetic nomenclature
5. **Abnormality Detection**: Trisomy (21, 18, 13), Monosomy X (Turner), translocations, deletions, inversions
6. **Downloadable Reports**: Text-based analysis reports
7. **YOLO Karyogram Generation**: ML-based pipeline (YOLOv8 detection + ResNet18/CNN classification) that generates visual karyogram images from metaphase spreads
8. **ML Training Pipeline**: Auto-annotation, dataset building, YOLO detector training, ResNet18 classifier training (two-phase fine-tuning with ImageNet pretrained backbone) for chromosome detection and classification
9. **Paired Training Pipeline**: Clinical workstation screenshot parsing, metaphase-karyogram pair matching, supervised training from ground-truth labeled data

## Use Cases

- Clinical cytogenetics labs analyzing metaphase spreads
- Research teams studying chromosomal abnormalities
- Educational demonstrations of karyotype analysis
- Rapid screening of chromosome images for common syndromes

## Mode
Single-product repository (product repo)
