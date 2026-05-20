"""Chromosome Karyotype Analyzer -- Streamlit entry point.

Thin orchestrator importing from cv/, vlm/, ui/ packages.
"""

import streamlit as st

from providers import APIProvider
from vlm import KaryotypeAnalyzer
from ui import (
    display_header,
    display_disclaimer,
    display_sidebar_settings,
    display_upload_section,
    display_analysis_section,
    display_results,
    display_consensus_results,
    display_two_stage_results,
    display_precision_lens_results,
    display_report_section,
)
from ui.styles import inject_css


def _init_session_state():
    """Initialize session state variables."""
    defaults = {
        "analysis_result": None,
        "uploaded_image": None,
        "raw_response": None,
        "consensus_api_keys": {"openai": None, "anthropic": None, "gemini": None},
        "consensus_settings": {
            "use_openai": True, "use_anthropic": True,
            "use_gemini": True, "min_agreement": "majority",
        },
        "cv_detection": None,
        "saved_api_key": None,
        "api_key_saved": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Chromosome Karyotype Analyzer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    _init_session_state()

    display_header()
    display_disclaimer()

    provider, api_key, consensus_keys = display_sidebar_settings()
    analyzer = KaryotypeAnalyzer(provider=provider, api_key=api_key)

    image = display_upload_section()

    if image is not None:
        if provider == APIProvider.YOLO_KARYOGRAM:
            from karyogram_ui import display_karyogram_analysis
            display_karyogram_analysis(image)
            return

        if provider == APIProvider.CONSENSUS:
            valid_keys = sum(1 for v in consensus_keys.values() if v)
            if valid_keys < 2:
                st.warning("Please provide API keys for at least 2 models.")
            else:
                display_analysis_section(analyzer, image, provider, consensus_keys)
        elif provider != APIProvider.MOCK and not api_key:
            st.warning(f"Please enter your {provider.value} API key in the sidebar.")
        else:
            display_analysis_section(analyzer, image, provider)

    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        if result.get("is_consensus"):
            display_consensus_results(result)
        elif result.get("pipeline") == "precision_lens":
            display_precision_lens_results(result)
        elif result.get("pipeline") == "two_stage":
            display_two_stage_results(result)
        else:
            display_results(result)
        display_report_section(result)

        if st.button("🔄 Start New Analysis"):
            st.session_state.analysis_result = None
            st.session_state.uploaded_image = None
            st.session_state.raw_response = None
            st.rerun()


if __name__ == "__main__":
    main()
