"""Analysis trigger section: button and spinner for running analysis."""

import streamlit as st
from typing import Dict, Optional
from PIL import Image

from providers import APIProvider


def display_analysis_section(analyzer, image: Image.Image, provider: APIProvider,
                             consensus_keys: Optional[Dict] = None):
    """Render the analysis trigger and progress section."""
    st.header("🔬 Chromosome Analysis")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        provider_name = provider.value
        button_text = f"🚀 Analyze with {provider_name}"

        if provider == APIProvider.MOCK:
            st.info("Demo Mode: Results will be simulated.")
        if provider == APIProvider.CV_VLM:
            st.info("**CV + VLM Hybrid Mode**: OpenCV counts, GPT-4 interprets.")
        if provider == APIProvider.PRECISION_LENS:
            st.info("**Precision Clinical Lens**: 6-stage pipeline. May take 2-3 minutes.")
        if provider == APIProvider.CONSENSUS:
            active = []
            if consensus_keys and consensus_keys.get("openai"):
                active.append("GPT-4")
            if consensus_keys and consensus_keys.get("anthropic"):
                active.append("Claude")
            if consensus_keys and consensus_keys.get("gemini"):
                active.append("Gemini")
            st.info(f"Consensus Voting: {', '.join(active)}")

        if st.button(button_text, type="primary", width="stretch"):
            if provider == APIProvider.CONSENSUS:
                spinner_text = "Running multi-model consensus... (2-3 min)"
            elif provider == APIProvider.PRECISION_LENS:
                spinner_text = "Precision Clinical Lens 6-stage analysis... (2-3 min)"
            else:
                spinner_text = f"Analyzing with {provider_name}... (up to 60s)"

            with st.spinner(spinner_text):
                try:
                    progress_bar = st.progress(0)
                    progress_bar.progress(10)
                    if provider == APIProvider.CONSENSUS:
                        result = analyzer.analyze(image, consensus_keys=consensus_keys)
                    else:
                        result = analyzer.analyze(image)
                    progress_bar.progress(100)
                    st.session_state.analysis_result = result
                    _show_completion_message(result)
                except Exception as e:
                    st.error("Analysis failed. Check your API key.")
                    with st.expander("Technical details", expanded=False):
                        st.code(str(e))


def _show_completion_message(result: Dict):
    if result.get("is_consensus"):
        agreement = result.get("agreement_level", 0)
        if agreement == 1.0:
            st.success("All models agree!")
        elif agreement >= 0.67:
            st.success("Majority agreement achieved!")
        else:
            st.warning("Models disagree, review carefully.")
        st.balloons()
    elif result.get("pipeline") == "precision_lens":
        st.success("Precision Clinical Lens analysis complete!")
        st.balloons()
    elif result.get("pipeline") == "two_stage":
        cv_count = result.get("cv_detection", {}).get("count", 0)
        st.success(f"Two-Stage complete - CV detected {cv_count} chromosomes!")
        st.balloons()
    elif result.get("confidence", 0) > 0:
        st.success("Analysis completed successfully!")
        st.balloons()
    else:
        st.warning("Analysis completed with issues.")
