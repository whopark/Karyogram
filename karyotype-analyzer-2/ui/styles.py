"""CSS styles injection for the Streamlit application."""

import streamlit as st


def inject_css():
    """Inject custom CSS styles into the page."""
    st.markdown("""
<style>
    .main-header {
        background-color: #3B82F6;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .disclaimer {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .notation-display {
        background-color: #1F2937;
        color: #F9FAFB;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .api-available {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .api-unavailable {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .consensus-agree {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .consensus-partial {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .consensus-disagree {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .voting-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .voting-table th, .voting-table td {
        border: 1px solid #E5E7EB;
        padding: 0.75rem;
        text-align: center;
    }
    .voting-table th {
        background-color: #F3F4F6;
    }
    .vote-winner {
        background-color: #D1FAE5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
