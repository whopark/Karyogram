# SPEC-KARYO-001 Acceptance Criteria

## Scenarios

### S1: Happy path -- Normal karyotype metaphase to karyogram
Given the Streamlit app is running with PyTorch and ultralytics installed
And YOLO detector weights exist at runs/detect/training/runs/chromosome_v2/weights/best.pt
And CNN classifier weights exist at training/models/chromosome_classifier.pth
And the user has selected "YOLO Karyogram (ML Pipeline)" from the sidebar
When the user uploads a metaphase spread PNG image of a normal 46,XX or 46,XY karyotype
And clicks the analysis button
Then the system displays a karyogram image in the main content area
And the karyogram image contains chromosome crops arranged in 7 Denver group rows
And each chromosome pair is labeled with its class number (1-22, X, Y)
And the ISCN notation displays as "46,XX" or "46,XY"
And the chromosome count displays as 46
And a download button for the karyogram PNG is present

### S2: YOLO detection produces bounding boxes
Given a metaphase spread image with visible chromosomes
When the YOLO detector processes the image with confidence threshold 0.25
Then the detector returns a list of bounding boxes
And each bounding box has x, y, width, height values within the image dimensions
And boxes with width < 4 or height < 4 are filtered out
And the returned count is between 30 and 50 for a typical metaphase spread

### S3: CNN classification assigns 24-class labels with pair refinement
Given 46 cropped chromosome images from YOLO detection
When the CNN classifier processes all crops in a batch
Then each crop receives one of 24 class labels (chr1-chr22, X, Y)
And softmax confidence scores are between 0.0 and 1.0
And after pair-based refinement, each autosome class (chr1-chr22) has exactly 2 assignments
And sex chromosomes have either XX (2 X, 0 Y) or XY (1 X, 1 Y) configuration

Oracle: Given a synthetic test input of 46 uniform-sized crops labeled by the CNN as [chr1, chr1, chr2, chr2, ..., chr22, chr22, X, Y], after pair refinement the count per autosome is exactly {chr1: 2, chr2: 2, ..., chr22: 2} and sex is {X: 1, Y: 1}, producing ISCN "46,XY".

### S4: Karyogram grid layout matches Denver group standard
Given classified chromosomes with the following distribution: chr1-22 (2 each), X (1), Y (1)
When the karyogram generator renders the grid image
Then Row 1 contains chr1, chr2, chr3, chr4, chr5 with a group separator between chr3 and chr4
And Row 2 contains chr6, chr7, chr8, chr9
And Row 3 contains chr10, chr11, chr12
And Row 4 contains chr13, chr14, chr15
And Row 5 contains chr16, chr17, chr18
And Row 6 contains chr19, chr20
And Row 7 contains chr21, chr22, X, Y with a group separator between chr22 and X
And each chromosome number appears with its homologous pair side by side

Oracle: Given exactly 2 copies each of chr1-chr22, 1 X, 1 Y (total 46), the rendered grid has 7 rows with chromosome counts per row: [10, 8, 6, 6, 6, 4, 6] where the count includes both copies of each pair. Row 1 has 5 pairs (A: 3 pairs + B: 2 pairs = 10 chromosomes). Row 7 has 2 pairs + 1 X + 1 Y = 6 chromosomes.

### S5: ISCN notation derivation for abnormal karyotypes
Given classified chromosomes with 47 total: chr1-22 (2 each), chr21 (extra copy, 3 total), X (2), Y (0)
When the karyotype builder derives the notation
Then the ISCN notation is "47,XX,+21"
And the sex chromosomes field is "XX"
And the abnormalities list contains "+21"
And the total chromosome count is 47

Oracle: Input labels = [chr1, chr1, chr2, chr2, ..., chr21, chr21, chr21, chr22, chr22, X, X]. Expected output: notation="47,XX,+21", sex="XX", count=47, abnormalities=["+21"].

### S6: YOLO_KARYOGRAM appears in sidebar when dependencies met
Given PyTorch (torch) is importable
And ultralytics is importable
When the Streamlit application loads and renders the sidebar
Then "YOLO Karyogram (ML Pipeline)" appears in the provider selection dropdown
And the APIProvider enum contains YOLO_KARYOGRAM

### S7: Karyogram results display includes all required components
Given a successful karyogram pipeline run producing a result dict and karyogram image
When display_karyogram_results renders the output
Then the Streamlit page shows:
And the karyogram image rendered via st.image
And the ISCN notation as text
And the total chromosome count
And the sex chromosome determination
And a list of detected abnormalities (or "None" if normal)
And a Denver group distribution summary
And a download button that produces a valid PNG file

### S8: Graceful degradation when dependencies are missing
Given PyTorch is NOT installed (torch import raises ImportError)
When the Streamlit application loads
Then "YOLO Karyogram (ML Pipeline)" does NOT appear in the sidebar dropdown
And all existing analysis modes (OpenAI, CV+VLM, Precision Lens, Demo) remain available
And no import error or crash occurs

### S9: Graceful degradation when weights are missing
Given PyTorch and ultralytics are installed
And the user selects YOLO_KARYOGRAM mode
But YOLO weights file does not exist at the configured path
When the user triggers analysis
Then the system displays an error message containing "weights not found"
And the error message suggests how to train or obtain the weights
And the application does not crash or show a Python traceback

### S10: File size compliance
Given the implementation is complete
When measuring line counts of new source files
Then ml_pipeline.py has at most 300 lines
And karyogram_generator.py has at most 300 lines
And karyogram_ui.py has at most 300 lines
