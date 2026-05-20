"""Common UI components: header, disclaimer, confidence class helper."""

import streamlit as st


def display_header():
    """Display the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Chromosome Karyotype Analyzer</h1>
        <p>AI-Powered Cytogenetic Analysis Tool</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">ISCN 2020 Compliant • Multi-Provider AI • Educational & Research Use Only</p>
    </div>
    """, unsafe_allow_html=True)


def display_disclaimer():
    """Display medical disclaimer."""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only.
        Results must be validated by qualified cytogenetics professionals. Do not use for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


def get_confidence_class(confidence: float) -> str:
    """Return CSS class name based on confidence score."""
    if confidence >= 85:
        return "confidence-high"
    elif confidence >= 65:
        return "confidence-medium"
    return "confidence-low"
