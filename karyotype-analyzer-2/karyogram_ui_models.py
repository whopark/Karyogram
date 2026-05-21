"""
Model loading and progress utilities for the YOLO Karyogram UI module.

Extracted from karyogram_ui.py to keep each module under 300 lines.
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Model loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_models(_cache_key: str = "v4") -> tuple:
    """Load YOLO detector and CNN classifier. Cached across reruns.

    Returns:
        Tuple of (detector_result, classifier_result) dicts.
    """
    from ml_pipeline import load_detector, load_classifier  # noqa: PLC0415
    det = load_detector()
    cls = load_classifier()
    # Do not cache error results — clear on next rerun
    # @AX:WARN: [AUTO] global cache side effect — st.cache_resource.clear() evicts ALL cached resources app-wide, not just models; may cause unexpected reloads of unrelated cached objects
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
# Download helper (moved from karyogram_ui.py for line budget)
# ---------------------------------------------------------------------------

def display_karyogram_download(karyogram_image) -> None:
    """Render a download button for the generated karyogram PNG."""
    import io as _io  # noqa: PLC0415
    buf = _io.BytesIO()
    karyogram_image.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="Download Karyogram PNG",
        data=buf,
        file_name="karyogram.png",
        mime="image/png",
    )
