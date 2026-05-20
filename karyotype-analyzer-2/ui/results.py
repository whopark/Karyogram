"""Standard and consensus result display components."""

import streamlit as st
from typing import Dict

from ui.common import get_confidence_class


def display_results(result: Dict):
    """Display standard single-provider analysis results."""
    st.header("📊 Analysis Results")
    st.caption(f"🤖 Analyzed by: **{result.get('provider', 'Unknown')}**")

    st.markdown(f'<div class="notation-display">{result.get("notation", "N/A")}</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chromosome Count", result.get("chromosome_count", "N/A"))
    with col2:
        st.metric("Sex Chromosomes", result.get("sex_chromosomes", "N/A"))
    with col3:
        conf = result.get("confidence", 0)
        cls = get_confidence_class(conf)
        st.markdown(f'<div style="text-align:center;"><h4>Confidence</h4>'
                    f'<p class="{cls}" style="font-size:2rem;">{conf:.1f}%</p></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get("abnormalities", [])
    if abnormalities:
        for i, ab in enumerate(abnormalities):
            st.write(f"{i+1}. **{ab.get('type','unknown').title()}** "
                     f"(Chr {ab.get('chromosome','N/A')}): {ab.get('description','')}")
    else:
        st.write("✓ No chromosomal abnormalities detected.")

    st.subheader("📝 Clinical Interpretation")
    st.write(result.get("interpretation", "No interpretation available."))
    if result.get("detailed_findings"):
        st.subheader("🔎 Detailed Findings")
        st.write(result.get("detailed_findings"))
    st.subheader("🔧 Technical Notes")
    st.write(result.get("technical_notes", "N/A"))
    st.caption(f"Analysis performed at: {result.get('analysis_time', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.raw_response:
        with st.expander("📄 View Raw API Response"):
            st.code(st.session_state.raw_response, language="json")


def display_consensus_results(result: Dict):
    """Display consensus voting results with breakdown."""
    st.header("🗳️ Multi-Model Consensus Results")
    agreement = result.get("agreement_level", 0)
    providers_used = result.get("providers_used", [])
    total = len(providers_used)

    if agreement == 1.0:
        cls, icon, text = "consensus-agree", "✅", f"All {total} models agree!"
    elif agreement >= 0.67:
        cls, icon, text = "consensus-partial", "🟡", f"Majority ({int(agreement*total)}/{total})"
    else:
        cls, icon, text = "consensus-disagree", "⚠️", "Models disagree"

    st.markdown(f'<div class="{cls}"><h3>{icon} {text}</h3>'
                f'<p>Providers: {", ".join(providers_used)}</p></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="notation-display"><strong>Consensus:</strong> '
                f'{result.get("notation","N/A")}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Count", result.get("chromosome_count", "N/A"))
    with c2: st.metric("Sex", result.get("sex_chromosomes", "N/A"))
    with c3:
        conf = result.get("confidence", 0)
        st.markdown(f'<div style="text-align:center;"><h4>Confidence</h4>'
                    f'<p class="{get_confidence_class(conf)}" style="font-size:1.5rem;">'
                    f'{conf:.1f}%</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div style="text-align:center;"><h4>Agreement</h4>'
                    f'<p style="font-size:1.5rem;font-weight:bold;">{agreement:.0%}</p></div>',
                    unsafe_allow_html=True)

    individual = result.get("individual_results", [])
    if individual:
        st.subheader("🗳️ Voting Breakdown")
        import pandas as pd
        rows = [{"Model": providers_used[i] if i < len(providers_used) else "?",
                 "Notation": r.get("notation", "N/A"),
                 "Count": r.get("chromosome_count", "N/A"),
                 "Sex": r.get("sex_chromosomes", "N/A"),
                 "Confidence": f"{r.get('confidence', 0):.1f}%"}
                for i, r in enumerate(individual)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get("abnormalities", [])
    if abnormalities:
        for i, ab in enumerate(abnormalities):
            st.markdown(f"**{i+1}. {ab.get('type','').title()}** (Chr {ab.get('chromosome','N/A')})\n"
                        f"- {ab.get('description','')}\n"
                        f"- *Agreement: {ab.get('agreement','N/A')}*")
    else:
        st.success("✓ No abnormalities detected by consensus.")

    st.subheader("📋 Individual Model Results")
    for i, r in enumerate(individual):
        pname = providers_used[i] if i < len(providers_used) else "Unknown"
        with st.expander(f"🤖 {pname}: {r.get('notation','N/A')}"):
            st.write(f"Count: {r.get('chromosome_count','N/A')}, "
                     f"Sex: {r.get('sex_chromosomes','N/A')}, "
                     f"Confidence: {r.get('confidence',0):.1f}%")
            st.write(r.get("interpretation", ""))

    errors = result.get("errors", [])
    if errors:
        with st.expander("⚠️ Provider Errors"):
            for e in errors:
                st.warning(e)
