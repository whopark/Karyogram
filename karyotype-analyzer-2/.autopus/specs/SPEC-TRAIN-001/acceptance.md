# SPEC-TRAIN-001 Acceptance Criteria

## Scenarios

### S1: Workstation screenshot split detection
Given a karyogram workstation screenshot `26-k-0603.png` (1800x950px RGBA, dark background, left=metaphase, right=arranged karyogram)
When `workstation_parser.detect_split_boundary()` is called on the grayscale conversion
Then the detected split x-coordinate is within 10% of the image width from the true visual boundary (approximately x=800-1000 for a 1800px-wide image).
And the returned karyogram crop contains only the right portion with the arranged chromosome grid.

### S2: Dark background inversion
Given a cropped karyogram region with median pixel intensity below 128 (dark background)
When `workstation_parser.invert_if_dark()` is applied
Then the output has median intensity above 128.
And when the inverted image is passed to `karyogram_parser.parse_karyogram()`, it detects between 30 and 55 chromosome contours (matching the expected range for a normal or abnormal karyotype).

### S3: Filename matching with zero-padding normalization
Given `metaphase/` containing `26-k-0659.png` and `karyogram/` containing `26-k-659.png`
When `pair_trainer.normalize_stem("26-k-0659")` and `pair_trainer.normalize_stem("26-k-659")` are computed
Then both return the identical normalized string `"26-k-659"`.
And `find_matched_pairs()` includes this as a valid matched pair.

### S4: Unmatched files are reported
Given `metaphase/` contains `26-k-0668.png` with no corresponding file in `karyogram/`
When `find_matched_pairs()` runs
Then stderr contains a warning line mentioning `26-k-0668.png` as unmatched.
And the unmatched file does not cause the pipeline to abort.

### S5: Labeled crops follow existing directory layout
Given workstation parser successfully extracts chromosomes from `26-k-0603.png`
When crops are saved to the output directory
Then the directory structure is `chromosome_crops/chr{N}/*.png` where N is in {1..22, X, Y}.
And each crop filename contains the source image stem (e.g., `26-k-0603_chr1_a.png`).
And `ChromosomeDataset("chromosome_crops/")` from `train_classifier.py` loads the crops without error and reports at least 1 sample per class for classes present in the karyogram.

### S6: YOLO annotations are generated for metaphase images
Given `metaphase/26-k-0603.png` exists
When `auto_annotate.process_image()` is called on it
Then `annotations/labels/26-k-0603.txt` is created.
And each line in the label file has format `0 cx cy w h` with all values in [0.0, 1.0].
And the number of lines (detected chromosomes) is between 20 and 60.

### S7: End-to-end pipeline produces trained models
Given the 8 matched pairs from `metaphase/` and `karyogram/`
When `python pair_trainer.py --metaphase_dir ../metaphase --karyogram_dir ../karyogram` runs to completion
Then `training/models/chromosome_classifier.pth` is created or updated.
And stdout contains lines matching `val top-1 \d+\.\d+` (classifier accuracy report).
And if ultralytics is installed, `runs/detect/` contains detector weights and stdout contains `mAP` metrics.

### S8: Metrics summary is printed (oracle acceptance)
Given the pipeline completes training on 8 matched pairs
When the final summary is printed to stdout
Then the summary contains exactly these fields with concrete numeric values (not "N/A" for classifier metrics):
  - "Matched pairs: 8"
  - "Total crops:" followed by an integer >= 300 (8 karyograms x ~44 chromosomes = ~352 expected)
  - "Classifier top-1:" followed by a float in [0.0, 1.0]
  - "Classifier top-3:" followed by a float in [0.0, 1.0]
  - "Classifier top-3" value >= "Classifier top-1" value (top-3 is always >= top-1 by definition)

### S9: Abnormal chromosome count warning
Given a karyogram workstation screenshot where the karyogram half yields only 25 detected contours after inversion
When `workstation_parser.parse_workstation_image()` processes it
Then stderr contains a warning with the filename and the count 25.
And processing continues (no exception raised).
And any crops that were extracted are still saved.
