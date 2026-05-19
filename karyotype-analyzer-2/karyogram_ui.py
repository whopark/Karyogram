"""
Streamlit UI module for YOLO Karyogram analysis mode.

Handles display, progress, results, download, and error states
for the ML-based chromosome karyogram pipeline.
"""

import io
import streamlit as st
from PIL import Image


# ---------------------------------------------------------------------------
# Session state keys (prefixed to avoid collision with other app modules)
# ---------------------------------------------------------------------------
_KEY_RESULT = "karyogram_result"
_KEY_IMAGE = "karyogram_image"
_KEY_MODELS_LOADED = "karyogram_models_loaded"

# Denver group metadata: group label, chromosomes included, expected pair count
_DENVER_GROUPS = [
    ("A", [1, 2, 3], 6),
    ("B", [4, 5], 4),
    ("C", [6, 7, 8, 9, 10, 11, 12], 14),
    ("D", [13, 14, 15], 6),
    ("E", [16, 17, 18], 6),
    ("F", [19, 20], 4),
    ("G", [21, 22], 4),
]


# ---------------------------------------------------------------------------
# Model loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_models(_cache_key: str = "v2") -> tuple:
    """Load YOLO detector and CNN classifier. Cached across reruns.

    Returns:
        Tuple of (detector_result, classifier_result) dicts.
    """
    from ml_pipeline import load_detector, load_classifier  # noqa: PLC0415
    det = load_detector()
    cls = load_classifier()
    # Do not cache error results — clear on next rerun
    if "error" in det or "error" in cls:
        st.cache_resource.clear()
    return det, cls


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _display_progress(stage: str, progress: float) -> None:
    """Update progress bar and status text.

    Args:
        stage: Human-readable description of the current stage.
        progress: Float between 0.0 and 1.0.
    """
    st.session_state["_karyogram_progress_bar"].progress(progress)
    st.session_state["_karyogram_status_text"].text(stage)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def display_karyogram_analysis(image: Image.Image) -> None:
    """Main entry point called from app.py when YOLO_KARYOGRAM mode is selected.

    Orchestrates the full analysis flow:
      1. Show progress bar
      2. Load models (cached)
      3. Run ML pipeline
      4. Generate karyogram image
      5. Display results
      6. Offer download

    Args:
        image: PIL Image uploaded by the user.
    """
    st.subheader("YOLO Karyogram Analysis")

    # Check dependency availability before anything else
    try:
        import torch  # noqa: F401, PLC0415
        import ultralytics  # noqa: F401, PLC0415
    except ImportError as exc:
        display_karyogram_error({
            "type": "missing_dependencies",
            "message": str(exc),
        })
        return

    # Confidence threshold slider
    conf = st.slider(
        "Detection Confidence Threshold",
        min_value=0.01, max_value=0.50, value=0.25, step=0.01,
        help="Lower values detect more chromosomes but may include debris. "
             "Recommended: 0.25 for the paired-trained model.",
    )

    # Initialize progress widgets and store in session state so helpers can update them
    status_text = st.empty()
    progress_bar = st.progress(0.0)
    st.session_state["_karyogram_status_text"] = status_text
    st.session_state["_karyogram_progress_bar"] = progress_bar

    try:
        # Stage 1 — load models
        _display_progress("Loading models...", 0.1)
        detector, classifier = _load_models()
        st.session_state[_KEY_MODELS_LOADED] = True

        # Check for model loading errors
        if "error" in detector:
            display_karyogram_error({"type": "missing_weights", "message": detector["error"]})
            return
        if "error" in classifier:
            display_karyogram_error({"type": "missing_weights", "message": classifier["error"]})
            return

        # Stage 2 — detect chromosomes
        _display_progress(f"Detecting chromosomes (conf={conf:.2f})...", 0.35)
        from ml_pipeline import run_pipeline  # noqa: PLC0415
        result = run_pipeline(image, detector, classifier, conf=conf)

        # Validate detection count before continuing
        count = result.get("count", result.get("chromosome_count", 0))
        if "error" in result:
            display_karyogram_error({"type": "pipeline_error", "message": result["error"]})
            return
        if count == 0:
            display_karyogram_error({"type": "no_detections"})
            return
        if count < 20:
            display_karyogram_error({"type": "too_few", "count": count})
            # Continue — still render what was found

        # Stage 3 — classify
        _display_progress("Classifying chromosomes...", 0.60)

        # Stage 4 — generate karyogram image
        _display_progress("Generating karyogram...", 0.80)
        from karyogram_generator import generate_karyogram  # noqa: PLC0415
        karyogram_image, _meta = generate_karyogram(result.get("classifications", []), image)

        # Persist to session state
        st.session_state[_KEY_RESULT] = result
        st.session_state[_KEY_IMAGE] = karyogram_image

        _display_progress("Done.", 1.0)

    except FileNotFoundError as exc:
        display_karyogram_error({"type": "missing_weights", "message": str(exc)})
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")
        return
    finally:
        # Clean up progress widgets after completion or error
        progress_bar.empty()
        status_text.empty()

    # Render results
    result = st.session_state.get(_KEY_RESULT)
    karyogram_image = st.session_state.get(_KEY_IMAGE)
    if result and karyogram_image:
        display_karyogram_results(result, karyogram_image)
        display_karyogram_download(karyogram_image)


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def display_karyogram_results(result: dict, karyogram_image: Image.Image) -> None:
    """Display the karyogram image and analysis results.

    Layout:
    - Full-width karyogram image
    - Metrics row: chromosome count | sex chromosomes | ISCN notation
    - Abnormalities section (if any)
    - Denver group distribution table
    - Per-chromosome classification details (expandable)

    Args:
        result: Dict produced by ml_pipeline.run_pipeline.
        karyogram_image: PIL Image of the arranged karyogram.
    """
    # Karyogram image — full width
    st.image(karyogram_image, caption="Generated Karyogram", use_column_width=True)

    # Summary metrics
    col_count, col_sex, col_iscn = st.columns(3)
    col_count.metric("Chromosome Count", result.get("count", result.get("chromosome_count", "—")))
    col_sex.metric("Sex Chromosomes", result.get("sex", result.get("sex_chromosomes", "—")))
    col_iscn.metric("ISCN Notation", result.get("notation", result.get("iscn_notation", "—")))

    # Abnormalities
    abnormalities = result.get("abnormalities", [])
    if abnormalities:
        with st.container():
            st.warning("**Detected Abnormalities**")
            for item in abnormalities:
                st.write(f"- {item}")

    # Denver group distribution table
    st.subheader("Denver Group Distribution")
    denver = result.get("denver_groups", {})

    rows = []
    for group_label, chr_numbers, expected in _DENVER_GROUPS:
        found = denver.get(group_label, 0)
        status = "OK" if found == expected else ("Low" if found < expected else "High")
        rows.append({
            "Group": group_label,
            "Chromosomes": ", ".join(str(n) for n in chr_numbers),
            "Expected": expected,
            "Found": found,
            "Status": status,
        })

    rows.append({
        "Group": "Sex",
        "Chromosomes": "X, Y",
        "Expected": 2,
        "Found": denver.get("C_sex", 0) + denver.get("G_sex", 0),
        "Status": result.get("sex", "—"),
    })

    st.dataframe(rows, use_container_width=True)

    # Per-chromosome details (expandable)
    classifications = result.get("classifications", [])
    with st.expander("Per-Chromosome Classification Details"):
        if classifications:
            from collections import Counter
            label_counts = Counter(c.get("label", "") for c in classifications)
            detail_rows = [
                {"Chromosome": k, "Count": v}
                for k, v in sorted(label_counts.items(), key=lambda x: (len(x[0]), x[0]))
            ]
            st.dataframe(detail_rows, use_container_width=True)
        else:
            st.write("No per-chromosome data available.")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def display_karyogram_download(karyogram_image: Image.Image) -> None:
    """Render a download button for the generated karyogram PNG.

    Args:
        karyogram_image: PIL Image to be offered for download.
    """
    buf = io.BytesIO()
    karyogram_image.save(buf, format="PNG")
    buf.seek(0)

    st.download_button(
        label="Download Karyogram PNG",
        data=buf,
        file_name="karyogram.png",
        mime="image/png",
    )


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------

def display_karyogram_error(error_info: dict) -> None:
    """Display a user-friendly error or warning message.

    Handles four error types:
    - missing_dependencies: torch / ultralytics not installed
    - missing_weights: model weight files not found
    - no_detections: zero chromosomes detected
    - too_few: fewer than 20 chromosomes detected

    Args:
        error_info: Dict with at least a 'type' key.
    """
    error_type = error_info.get("type", "unknown")

    if error_type == "missing_dependencies":
        st.info(
            "Required ML dependencies are not installed. "
            "Run the following command and restart the app:\n\n"
            "```\npip install torch torchvision ultralytics\n```"
        )

    elif error_type == "missing_weights":
        msg = error_info.get("message", "")
        st.error(
            f"Model weight files not found. Details: `{msg}`\n\n"
            "Place weights in `weights/` or train: "
            "`python train.py --data karyotype.yaml --epochs 100`"
        )

    elif error_type == "pipeline_error":
        msg = error_info.get("message", "Unknown pipeline error")
        st.error(f"Analysis pipeline error: `{msg}`")

    elif error_type == "no_detections":
        st.warning(
            "No chromosomes detected in the image. "
            "Try adjusting the confidence threshold or use a higher-quality metaphase spread image."
        )

    elif error_type == "too_few":
        count = error_info.get("count", 0)
        st.warning(
            f"Only {count} chromosome(s) detected. Results may be unreliable. "
            "A normal human karyotype contains 46 chromosomes. "
            "Consider using a clearer image or lowering the detection threshold."
        )

    else:
        st.error(f"An unexpected error occurred: {error_info.get('message', 'Unknown error')}")
