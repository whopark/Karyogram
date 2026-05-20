"""Advanced result displays: Two-Stage pipeline and Precision Clinical Lens."""

import streamlit as st
from typing import Dict

from ui.common import get_confidence_class


def display_two_stage_results(result: Dict):
    """Display two-stage pipeline results with CV detection info."""
    st.header("🔬 Two-Stage Pipeline Results")
    st.info("📊 **Stage 1:** CV Detection → **Stage 2:** VLM Classification")

    cv = result.get("cv_detection", st.session_state.get("cv_detection") or {})
    if cv:
        st.subheader("🖥️ Stage 1: Computer Vision Detection")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("CV Count", cv.get("count", 0))
        with c2: st.metric("Sex Chr", cv.get("sex_chromosome_region", {}).get("estimated", "?"))
        with c3: st.metric("Method", cv.get("detection_method", "N/A").replace("_", " ").title())
        groups = cv.get("group_counts", {})
        if groups:
            with st.expander("📊 Denver Group Distribution"):
                import pandas as pd
                data = {"Group": ["A(1-3)", "B(4-5)", "C(6-12+X)", "D(13-15)", "E(16-18)", "F(19-20)", "G(21-22+Y)"],
                        "Count": [groups.get(g, 0) for g in "ABCDEFG"], "Expected": [6, 4, 15, 6, 6, 4, 5]}
                st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

    st.subheader("🤖 Stage 2: VLM Classification")
    st.markdown(f'<div class="notation-display">{result.get("notation","N/A")}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Final Count", result.get("chromosome_count", "N/A"))
    with c2: st.metric("Sex", result.get("sex_chromosomes", "N/A"))
    with c3:
        conf = result.get("confidence", 0)
        st.markdown(f'<div style="text-align:center;"><h4>Confidence</h4>'
                    f'<p class="{get_confidence_class(conf)}" style="font-size:2rem;">{conf:.1f}%</p></div>',
                    unsafe_allow_html=True)

    _show_abnormalities(result)
    st.subheader("📝 Clinical Interpretation")
    st.write(result.get("interpretation", "N/A"))
    if result.get("detailed_findings"):
        st.subheader("🔎 Detailed Findings")
        st.write(result["detailed_findings"])
    st.caption(f"Analysis at: {result.get('analysis_time','N/A')}")
    if st.session_state.raw_response:
        with st.expander("📄 Raw Response"):
            st.code(st.session_state.raw_response, language="json")


def display_precision_lens_results(result: Dict):
    """Display Precision Clinical Lens 6-stage pipeline results."""
    st.header("🔬 Precision Clinical Lens")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
                border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;color:white;text-align:center;">
        <h3 style="margin:0 0 1rem 0;color:#e0e0e0;">🔬 6-Stage Pipeline</h3>
        <div style="display:flex;justify-content:center;align-items:center;flex-wrap:wrap;gap:0.3rem;">
            <span style="background:#3B82F6;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.85rem;">1. Counting</span>
            <span style="color:#64748b;">→</span>
            <span style="background:#6366F1;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.85rem;">2. Classification</span>
            <span style="color:#64748b;">→</span>
            <span style="background:#8B5CF6;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.85rem;">3. Cluster</span>
            <span style="color:#64748b;">→</span>
            <span style="background:#A855F7;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.85rem;">4. Translocation</span>
            <span style="color:#64748b;">→</span>
            <span style="background:#D946EF;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.85rem;">5. Analysis</span>
            <span style="color:#64748b;">→</span>
            <span style="background:#EC4899;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.85rem;">6. Abnormality</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="notation-display" style="font-size:1.5rem;">'
                f'<strong>Final:</strong> {result.get("notation","N/A")}</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Count", result.get("chromosome_count", "N/A"))
    with c2: st.metric("Sex Chr", result.get("sex_chromosomes", "N/A"))
    with c3:
        conf = result.get("confidence", 0)
        st.markdown(f'<div style="text-align:center;"><h4>Confidence</h4>'
                    f'<p class="{get_confidence_class(conf)}" style="font-size:2rem;">{conf:.1f}%</p></div>',
                    unsafe_allow_html=True)

    stages = result.get("stages", {})
    from vlm.precision_lens import PrecisionClinicalLens
    for i, (kr, en) in enumerate(PrecisionClinicalLens.STAGE_NAMES):
        sd = stages.get(f"stage_{i+1}", {})
        with st.expander(f"Stage {i+1}: {en}", expanded=(i == 5)):
            if sd.get("parse_error"):
                st.warning("Could not parse this stage's response as JSON.")
                st.code(sd.get("raw_text", "No data"), language="text")
            else:
                _render_stage(i, sd, result)

    st.caption(f"Analysis at: {result.get('analysis_time','N/A')}")
    if st.session_state.raw_response:
        with st.expander("📄 Full Pipeline Data"):
            st.code(st.session_state.raw_response, language="json")


def _show_abnormalities(result: Dict):
    st.subheader("🔍 Detected Abnormalities")
    abnormalities = result.get("abnormalities", [])
    if abnormalities:
        for i, ab in enumerate(abnormalities):
            st.write(f"{i+1}. **{ab.get('type','').title()}** "
                     f"(Chr {ab.get('chromosome','N/A')}): {ab.get('description','')}")
    else:
        st.success("✓ No abnormalities detected.")


def _render_stage(idx: int, sd: Dict, result: Dict):
    """Render individual stage data inside an expander."""
    if idx == 0:
        st.metric("Total Count", sd.get("total_count", "N/A"))
        pos = sd.get("position_counts", {})
        if pos:
            auto = {k: v for k, v in pos.items() if k not in ("X", "Y")}
            cols = st.columns(6)
            for j, (k, v) in enumerate(auto.items()):
                with cols[j % 6]:
                    st.write(f"Chr {k}: {v} {'✅' if v == 2 else '⚠️'}")
            sex_p = {k: v for k, v in pos.items() if k in ("X", "Y")}
            if sex_p:
                st.write(f"**Sex:** X={sex_p.get('X','?')}, Y={sex_p.get('Y','?')}")
    elif idx == 1:
        groups = sd.get("denver_groups", {})
        if groups:
            import pandas as pd
            rows = [{"Group": g, "Count": d.get("count", 0), "Expected": d.get("expected", "?"),
                     "Chr": d.get("chromosomes", "")}
                    for g, d in groups.items() if isinstance(d, dict)]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    elif idx == 2:
        st.metric("Sex Determination", sd.get("sex_determination", "N/A"))
        anomalous = sd.get("anomalous_positions", [])
        if anomalous:
            st.warning(f"Anomalous positions: {', '.join(str(a) for a in anomalous)}")
        else:
            st.success("All positions normal")
    elif idx == 3:
        if sd.get("normal_structure", True):
            st.success("No structural abnormalities")
        else:
            for ab in sd.get("structural_abnormalities", []):
                st.error(f"**{ab.get('iscn_notation','N/A')}** - {ab.get('description','')}")
    elif idx == 4:
        st.markdown(f"**Preliminary:** `{sd.get('preliminary_karyotype','N/A')}`")
        cv = sd.get("cross_validation", {})
        if cv and (cv.get("count_consistent", True) and cv.get("groups_consistent", True)):
            st.success("Cross-validation passed")
        elif cv:
            st.warning(f"Discrepancies: {cv.get('discrepancies', [])}")
    elif idx == 5:
        abnormalities = sd.get("abnormalities", [])
        if abnormalities:
            for ab in abnormalities:
                st.write(f"**{ab.get('type','').title()} - {ab.get('subtype','')}** "
                         f"(Chr {ab.get('chromosome','N/A')}): {ab.get('description','')}")
                if ab.get("clinical_significance"):
                    st.write(f"   Clinical: _{ab['clinical_significance']}_")
        else:
            st.success("No abnormalities - normal karyotype")
        st.subheader("📝 Clinical Interpretation")
        st.write(sd.get("interpretation", result.get("interpretation", "N/A")))
        if sd.get("recommendations"):
            st.subheader("💡 Recommendations")
            st.write(sd["recommendations"])
