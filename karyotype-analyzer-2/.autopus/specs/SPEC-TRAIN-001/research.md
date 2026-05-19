# SPEC-TRAIN-001 Research

## Existing Code Analysis

### Existing training pipeline files

| File | Lines | Role | Key functions |
|------|-------|------|---------------|
| `training/karyogram_parser.py` | 260 | Extracts labeled chromosome crops | `parse_karyogram()`, `cluster_rows()`, `assign_labels()` |
| `training/auto_annotate.py` | 268 | YOLO bounding-box annotation | `process_image()`, `build_binary_masks()` |
| `training/build_dataset.py` | 291 | YOLO dataset builder with augmentation | `main(argv)` |
| `training/train_detector.py` | 252 | YOLOv8 fine-tuning | `main()` |
| `training/train_classifier.py` | 297 | ChromosomeCNN training | `train(args)`, `ChromosomeDataset`, `evaluate()` |
| `training/predict.py` | 263 | End-to-end inference | `analyze_image()`, `_build_karyotype()` |

