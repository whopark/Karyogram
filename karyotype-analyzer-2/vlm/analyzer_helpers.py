"""KaryotypeAnalyzer helper functions: response parsing, consensus
calculation, CV-hint analysis, and interpretation generation."""

import streamlit as st
import io
import json
import re
import base64
from datetime import datetime
from collections import Counter
from typing import Dict, List
from PIL import Image

from providers import OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE, GEMINI_AVAILABLE, APIProvider

if OPENAI_AVAILABLE:
    from openai import OpenAI


def parse_response(raw_content: str, provider_name: str) -> Dict:
    """Parse API response extracting JSON from various formats."""
    if not raw_content or not raw_content.strip():
        return create_error_response("Empty response from API", "", provider_name)
    try:
        try:
            return finalize_result(json.loads(raw_content.strip()), provider_name)
        except json.JSONDecodeError:
            pass
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            m = re.search(pattern, raw_content)
            if m:
                try:
                    return finalize_result(json.loads(m.group(1).strip()), provider_name)
                except json.JSONDecodeError:
                    continue
        first_brace = raw_content.find("{")
        if first_brace != -1:
            brace_count = 0
            in_string = False
            escape_next = False
            end_pos = -1
            for i, char in enumerate(raw_content[first_brace:], start=first_brace):
                if escape_next:
                    escape_next = False; continue
                if char == "\\" and in_string:
                    escape_next = True; continue
                if char == '"' and not escape_next:
                    in_string = not in_string; continue
                if not in_string:
                    if char == "{": brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i; break
            if end_pos != -1:
                try:
                    return finalize_result(json.loads(raw_content[first_brace : end_pos + 1]), provider_name)
                except json.JSONDecodeError:
                    pass
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw_content)
        fb, lb = cleaned.find("{"), cleaned.rfind("}")
        if fb != -1 and lb != -1:
            try:
                return finalize_result(json.loads(cleaned[fb : lb + 1]), provider_name)
            except json.JSONDecodeError:
                pass
        raise ValueError("No valid JSON found in response")
    except Exception as e:
        return create_error_response(str(e), raw_content, provider_name)


def finalize_result(result: Dict, provider_name: str) -> Dict:
    result.setdefault("notation", "Unable to determine")
    result.setdefault("chromosome_count", 0)
    result.setdefault("sex_chromosomes", "Unknown")
    result.setdefault("abnormalities", [])
    result.setdefault("confidence", 0)
    result.setdefault("interpretation", "Analysis incomplete")
    result.setdefault("detailed_findings", "")
    result["analysis_time"] = datetime.now().isoformat()
    result["technical_notes"] = f"Analysis performed using {provider_name}"
    result["provider"] = provider_name
    return result


def create_error_response(error_msg: str, raw_content: str, provider_name: str) -> Dict:
    preview = raw_content[:500] if raw_content else "No response content"
    return {
        "notation": "Parse Error", "chromosome_count": 0, "sex_chromosomes": "Unknown",
        "abnormalities": [], "confidence": 0,
        "interpretation": f"Failed to parse: {error_msg}",
        "detailed_findings": f"Raw: {preview}...",
        "analysis_time": datetime.now().isoformat(),
        "technical_notes": f"Error parsing response from {provider_name}.",
        "provider": provider_name,
    }


def generate_interpretation(count: int, abnormalities: list, sex: str) -> str:
    if count == 46 and not abnormalities:
        gender = "female" if sex == "XX" else "male"
        return f"Normal {gender} karyotype with no apparent abnormalities."
    parts = []
    if count != 46:
        parts.append(f"Numerical abnormality: {count} chromosomes detected.")
    for ab in abnormalities:
        t, c = ab.get("type", ""), ab.get("chromosome", "")
        if t == "trisomy":
            syndromes = {"21": "Down syndrome", "18": "Edwards syndrome", "13": "Patau syndrome"}
            parts.append(syndromes.get(c, f"Trisomy {c} detected."))
        elif t == "monosomy" and c == "X":
            parts.append("Turner syndrome (45,X).")
        elif t == "translocation":
            parts.append(f"Translocation {ab.get('description', '')} detected.")
    return " ".join(parts)


def calculate_consensus(analyzer, image: Image.Image, api_keys: Dict) -> Dict:
    """Run analysis on multiple providers and vote on results."""
    results, errors, providers_used = [], [], []
    configs = [
        ("openai", APIProvider.OPENAI, OPENAI_AVAILABLE, "GPT-4 Vision"),
        ("anthropic", APIProvider.ANTHROPIC, ANTHROPIC_AVAILABLE, "Claude Vision"),
        ("gemini", APIProvider.GEMINI, GEMINI_AVAILABLE, "Gemini Vision"),
    ]
    settings = st.session_state.consensus_settings
    for key_name, penum, avail, dname in configs:
        if not settings.get(f"use_{key_name}", True): continue
        if not avail: errors.append(f"{dname}: Package not installed"); continue
        if not api_keys.get(key_name): errors.append(f"{dname}: Key not provided"); continue
        try:
            from vlm.analyzer import KaryotypeAnalyzer
            tmp = KaryotypeAnalyzer(provider=penum, api_key=api_keys[key_name])
            if penum == APIProvider.OPENAI: r = tmp._analyze_with_openai(image)
            elif penum == APIProvider.ANTHROPIC: r = tmp._analyze_with_anthropic(image)
            elif penum == APIProvider.GEMINI: r = tmp._analyze_with_gemini(image)
            else: continue
            results.append(r); providers_used.append(dname)
        except Exception as e:
            errors.append(f"{dname}: {e}")
    if not results:
        return {"notation": "No Results", "chromosome_count": 0, "sex_chromosomes": "Unknown",
                "abnormalities": [], "confidence": 0, "interpretation": "No successful analyses.",
                "analysis_time": datetime.now().isoformat(), "provider": "Multi-Model Consensus",
                "is_consensus": True, "individual_results": [], "agreement_level": 0, "errors": errors}
    return _vote(results, providers_used, errors)


def _vote(results: List[Dict], providers: List[str], errors: List[str]) -> Dict:
    total = len(results)
    if total == 1:
        s = results[0]; s.update({"is_consensus": True, "individual_results": results,
            "agreement_level": 1.0, "providers_used": providers,
            "voting_breakdown": {"chromosome_count": {s["chromosome_count"]: 1},
                                 "sex_chromosomes": {s["sex_chromosomes"]: 1},
                                 "notation": {s["notation"]: 1}},
            "provider": f"Consensus ({providers[0]} only)", "errors": errors})
        return s
    cnt_v = Counter(r.get("chromosome_count", 0) for r in results)
    sex_v = Counter(r.get("sex_chromosomes", "Unknown") for r in results)
    not_v = Counter(r.get("notation", "Unknown") for r in results)
    c_cnt, c_agr = cnt_v.most_common(1)[0]
    c_sex, _ = sex_v.most_common(1)[0]
    c_not, _ = not_v.most_common(1)[0]
    agreement = c_agr / total
    abnorm_counts: Dict = {}
    for r in results:
        for ab in r.get("abnormalities", []):
            key = f"{ab.get('type', '')}:{ab.get('chromosome', '')}"
            if key not in abnorm_counts:
                abnorm_counts[key] = {"abnormality": ab, "detected_by": [], "count": 0}
            abnorm_counts[key]["count"] += 1
            abnorm_counts[key]["detected_by"].append(r.get("provider", "Unknown"))
    merged_ab = []
    for data in abnorm_counts.values():
        ab = data["abnormality"].copy()
        ab["agreement"] = f"{data['count']}/{total} models"
        ab["detected_by"] = data["detected_by"]
        merged_ab.append(ab)
    base_conf = sum(r.get("confidence", 0) for r in results) / total
    final_conf = min(99, base_conf + agreement * 10)
    if agreement == 1.0: at = "All models unanimously agree."
    elif agreement >= 0.67: at = f"Majority agreement ({c_agr}/{total} models)."
    else: at = "Models disagree."
    interps = [r.get("interpretation", "") for r in results if r.get("interpretation")]
    ci = f"{at}\n\n" + "\n---\n".join(f"**{providers[i]}**: {interps[i]}" for i in range(len(interps)))
    return {
        "notation": c_not, "chromosome_count": c_cnt, "sex_chromosomes": c_sex,
        "abnormalities": merged_ab, "confidence": round(final_conf, 1),
        "interpretation": ci, "detailed_findings": f"Consensus from {total} models",
        "analysis_time": datetime.now().isoformat(),
        "technical_notes": f"Multi-model consensus with {agreement:.0%} agreement",
        "provider": "Multi-Model Consensus", "is_consensus": True,
        "individual_results": results, "agreement_level": agreement,
        "providers_used": providers, "voting_breakdown": {
            "chromosome_count": dict(cnt_v), "sex_chromosomes": dict(sex_v), "notation": dict(not_v)},
        "errors": errors,
    }


def analyze_with_cv_hints(api_key: str, image: Image.Image, pos21: int, sex_est: str, sex_count: int = 0) -> Dict:
    """VLM analysis with CV position hints when total count is unreliable."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not installed")
    buffered = io.BytesIO()
    img = image.convert("RGB") if image.mode != "RGB" else image
    img.save(buffered, format="JPEG", quality=95)
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    triple_x_warning = ""
    if pos21 >= 3 and sex_count <= 1:
        triple_x_warning = ("\nWARNING: CV detected 3 near position 21 but few in sex region. "
                            "Check if they are actually three X chromosomes (Triple X, NOT Down syndrome).\n")
    prompt = f"""You are an expert clinical cytogeneticist.
## CV HINTS (verify visually)
- ~{pos21} chromosome(s) near position 21
- {sex_count} chromosome(s) in sex region
- CV estimated sex: {sex_est}
{triple_x_warning}
## ANALYSIS ORDER
1. COUNT SEX CHROMOSOMES FIRST
2. CHECK FOR Y CHROMOSOME
3. COUNT POSITION 21
Return ONLY valid JSON:
{{
    "notation": "ISCN notation",
    "chromosome_count": number,
    "sex_chromosomes": "XX/XY/X/XXY/XXX/XYY",
    "abnormalities": [{{"type": "type", "chromosome": "chr", "description": "desc"}}],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "detailed_findings": "X count, Y present/absent, position 21 count"
}}"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a board-certified clinical cytogeneticist. Count sex chromosomes FIRST. Return valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}},
            ]},
        ],
        max_tokens=1500, temperature=0.1,
    )
    raw = response.choices[0].message.content
    st.session_state.raw_response = raw
    return parse_response(raw, "CV+VLM (CV-assisted)")


def build_two_stage_prompt(cv_count, cv_groups, sex_info) -> str:
    """Build the two-stage VLM prompt from CV detection results."""
    sex_est = sex_info.get("estimated", "unknown")
    return f"""You are an expert clinical cytogeneticist.
## CV DETECTION RESULTS (Stage 1)
**Detected Chromosome Count: {cv_count}**
Estimated Denver Group Distribution:
- Group A (1-3): {cv_groups.get('A', 0)}
- Group B (4-5): {cv_groups.get('B', 0)}
- Group C (6-12+X): {cv_groups.get('C', 0)}
- Group D (13-15): {cv_groups.get('D', 0)}
- Group E (16-18): {cv_groups.get('E', 0)}
- Group F (19-20): {cv_groups.get('F', 0)}
- Group G (21-22+Y): {cv_groups.get('G', 0)}
Sex Chromosome Region: {sex_est}
## YOUR TASK (Stage 2)
Based on the CV count of **{cv_count}**, determine the karyotype.
Return ONLY valid JSON:
{{
    "notation": "ISCN notation",
    "chromosome_count": {cv_count},
    "sex_chromosomes": "XX/XY/X/XXY/XXX",
    "cv_detection": {{"count": {cv_count}, "groups": {json.dumps(cv_groups)}, "sex_region": "{sex_est}"}},
    "abnormalities": [{{"type": "type", "chromosome": "chr", "description": "desc"}}],
    "confidence": number (0-100),
    "interpretation": "clinical interpretation",
    "detailed_findings": "how CV informed diagnosis"
}}"""
