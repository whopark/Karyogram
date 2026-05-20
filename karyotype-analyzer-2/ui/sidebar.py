"""Sidebar settings: provider selection, API key input, status display."""

import streamlit as st
from typing import Tuple, Dict, Optional

from providers import (
    APIProvider,
    OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE, GEMINI_AVAILABLE,
    CV2_AVAILABLE, TORCH_AVAILABLE, YOLO_AVAILABLE,
)
from ui.sidebar_helpers import inject_local_storage_script


def display_api_status():
    """Display API package status in sidebar."""
    st.sidebar.subheader("📦 Package Status")
    if OPENAI_AVAILABLE:
        html = '<div class="api-status api-available">✓ OpenAI GPT-4 Vision ready</div>'
    else:
        html = '<div class="api-status api-unavailable">✗ OpenAI package missing</div>'
    st.sidebar.markdown(html, unsafe_allow_html=True)


def display_sidebar_settings() -> Tuple[APIProvider, Optional[str], Dict]:
    """Render sidebar settings and return (provider, api_key, consensus_keys)."""
    with st.sidebar:
        st.header("⚙️ Settings")
        display_api_status()
        st.divider()
        st.subheader("🔌 API Provider")

        provider_options = ["Demo Mode (No API)"]
        if OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE or GEMINI_AVAILABLE:
            provider_options.insert(0, "Precision Clinical Lens (6-Stage)")
        if OPENAI_AVAILABLE:
            provider_options.insert(0, "OpenAI GPT-4 Vision")
            if CV2_AVAILABLE:
                provider_options.insert(1, "CV + VLM (Hybrid)")
        if TORCH_AVAILABLE and YOLO_AVAILABLE:
            provider_options.insert(0, "YOLO Karyogram (ML Pipeline)")

        selected = st.selectbox("Select AI Provider", provider_options,
                                help="CV+VLM uses computer vision for counting, VLM for interpretation")

        provider_map = {
            "YOLO Karyogram (ML Pipeline)": APIProvider.YOLO_KARYOGRAM,
            "OpenAI GPT-4 Vision": APIProvider.OPENAI,
            "CV + VLM (Hybrid)": APIProvider.CV_VLM,
            "Precision Clinical Lens (6-Stage)": APIProvider.PRECISION_LENS,
            "Demo Mode (No API)": APIProvider.MOCK,
        }
        provider = provider_map.get(selected, APIProvider.MOCK)

        if provider == APIProvider.CV_VLM:
            st.info("**CV + VLM Mode**: OpenCV counts chromosomes, then GPT-4 interprets.")
        if provider == APIProvider.PRECISION_LENS:
            st.info("**Precision Clinical Lens**: 6-stage sequential analysis pipeline.")

        st.divider()

        api_key = None
        consensus_keys: Dict = {}

        if provider in [APIProvider.OPENAI, APIProvider.CV_VLM, APIProvider.PRECISION_LENS]:
            default_key = st.session_state.get("saved_api_key", "") or ""
            api_key = st.text_input("OpenAI API Key", value=default_key, type="password",
                                    help="Enter your OpenAI API key (starts with sk-)",
                                    key="openai_api_key_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", use_container_width=True):
                    clean = api_key.strip() if api_key else ""
                    valid = clean.startswith("sk-") and len(clean) >= 20 and all(c.isalnum() or c in "-_" for c in clean)
                    if valid:
                        st.session_state.saved_api_key = clean
                        st.session_state.api_key_saved = True
                        st.markdown(inject_local_storage_script("save", clean), unsafe_allow_html=True)
                        st.success("Saved!")
                    else:
                        st.error("Invalid key format")
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.saved_api_key = None
                    st.session_state.api_key_saved = False
                    st.markdown(inject_local_storage_script("clear"), unsafe_allow_html=True)
                    st.info("Cleared!")
                    st.rerun()
            if st.session_state.get("saved_api_key"):
                st.caption("🔐 API key saved (persists in browser)")
            else:
                st.caption("Get key: [platform.openai.com](https://platform.openai.com/api-keys)")
            st.markdown(inject_local_storage_script("load"), unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        ### About
        This tool analyzes metaphase spreads and generates ISCN 2020 karyotype notations.
        ### Resources
        - [ISCN 2020](https://karger.com/books/book/367/An-International-System-for-Human-Cytogenomic)
        - [Karyotype Guide](https://www.ncbi.nlm.nih.gov/books/NBK557817/)
        """)
        return provider, api_key, consensus_keys
