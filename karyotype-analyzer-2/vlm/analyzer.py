"""KaryotypeAnalyzer: routes analysis to provider-specific VLM implementations."""

import streamlit as st
import io
import json
import re
import base64
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image

from providers import (
    APIProvider,
    OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE, GEMINI_AVAILABLE,
)

if OPENAI_AVAILABLE:
    from openai import OpenAI
if ANTHROPIC_AVAILABLE:
    import anthropic
if GEMINI_AVAILABLE:
    from google import genai

from vlm.prompts import KARYOTYPE_ANALYSIS_PROMPT, CV_VLM_INTERPRETATION_PROMPT
from vlm.analyzer_helpers import (
    parse_response,
    finalize_result,
    create_error_response,
    generate_interpretation,
    calculate_consensus,
    analyze_with_cv_hints,
    build_two_stage_prompt,
)


class KaryotypeAnalyzer:
    """Multi-provider karyotype analyzer dispatching to VLM backends."""

    def __init__(self, provider: APIProvider, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key

    def encode_image_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    def analyze(self, image: Image.Image, consensus_keys: Optional[Dict] = None) -> Dict:
        dispatch = {
            APIProvider.OPENAI: self._analyze_with_openai,
            APIProvider.ANTHROPIC: self._analyze_with_anthropic,
            APIProvider.GEMINI: self._analyze_with_gemini,
            APIProvider.PRECISION_LENS: self._analyze_with_precision_lens,
            APIProvider.TWO_STAGE: self._analyze_with_two_stage,
            APIProvider.CV_VLM: self._analyze_with_cv_vlm,
        }
        if self.provider == APIProvider.CONSENSUS:
            return self._analyze_with_consensus(image, consensus_keys or {})
        fn = dispatch.get(self.provider)
        if fn:
            return fn(image)
        return self._mock_analysis()

    # -- Single-model providers -----------------------------------------

    def _analyze_with_openai(self, image: Image.Image) -> Dict:
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed.")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        client = OpenAI(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a board-certified clinical cytogeneticist. CRITICAL: COUNT chromosomes accurately. Always provide results as valid JSON."},
                {"role": "user", "content": [
                    {"type": "text", "text": KARYOTYPE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}},
                ]},
            ],
            max_tokens=2500, temperature=0.1,
        )
        raw = response.choices[0].message.content
        st.session_state.raw_response = raw
        return parse_response(raw, "OpenAI GPT-4 Vision")

    def _analyze_with_anthropic(self, image: Image.Image) -> Dict:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed.")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        client = anthropic.Anthropic(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=2500,
            system="You are a board-certified clinical cytogeneticist. CRITICAL: COUNT chromosomes accurately. Always provide results as valid JSON.",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                {"type": "text", "text": KARYOTYPE_ANALYSIS_PROMPT},
            ]}],
        )
        raw = response.content[0].text
        st.session_state.raw_response = raw
        return parse_response(raw, "Anthropic Claude Vision")

    def _analyze_with_gemini(self, image: Image.Image) -> Dict:
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI package not installed.")
        if not self.api_key:
            raise ValueError("Google API key is required")
        client = genai.Client(api_key=self.api_key)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        system_instruction = "CRITICAL: COUNT chromosomes accurately. Always provide results as valid JSON.\n\n"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[system_instruction + KARYOTYPE_ANALYSIS_PROMPT, image],
            config={"temperature": 0.1, "max_output_tokens": 2500},
        )
        raw = response.text
        st.session_state.raw_response = raw
        return parse_response(raw, "Google Gemini Vision")

    # -- Hybrid modes ---------------------------------------------------

    def _analyze_with_two_stage(self, image: Image.Image) -> Dict:
        if not CV2_AVAILABLE_CHECK():
            raise ImportError("OpenCV not installed.")
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed.")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        from cv.detector import ChromosomeDetector
        detector = ChromosomeDetector()
        detection = detector.detect_chromosomes(image)
        st.session_state.cv_detection = detection
        cv_count = detection.get("count", 0)
        cv_groups = detection.get("group_counts", {})
        sex_info = detection.get("sex_chromosome_region", {})
        prompt = build_two_stage_prompt(cv_count, cv_groups, sex_info)
        client = OpenAI(api_key=self.api_key)
        image_base64 = self.encode_image_base64(image)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a cytogeneticist. A CV system detected {cv_count} chromosomes. Trust the CV detection."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}},
                ]},
            ],
            max_tokens=2500, temperature=0.1,
        )
        raw = response.choices[0].message.content
        st.session_state.raw_response = raw
        result = parse_response(raw, "Two-Stage Pipeline (CV + VLM)")
        result["cv_detection"] = detection
        result["pipeline"] = "two_stage"
        return result

    def _analyze_with_cv_vlm(self, image: Image.Image) -> Dict:
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed.")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not CV2_AVAILABLE_CHECK():
            raise ImportError("OpenCV not available.")
        from cv.detector import ChromosomeDetector
        detector = ChromosomeDetector()
        cv_result = detector.detect_karyogram_positions(image)
        st.session_state.cv_detection = cv_result
        total_count = cv_result.get("total_count", 0)
        pos_counts = cv_result.get("position_counts", {})
        pos21 = pos_counts.get("position_21", 0)
        sex_count = pos_counts.get("sex_chromosomes", 0)
        sex_est = pos_counts.get("sex_chr_estimated", "unknown")
        ka = cv_result.get("karyotype_analysis", {})

        if total_count < 44 or total_count > 50:
            st.warning(f"CV detected {total_count} chromosomes (outside 44-50). Using VLM with CV hints.")
            result = analyze_with_cv_hints(self.api_key, image, pos21, sex_est, sex_count)
            result["fallback_used"] = True
            result["cv_detection"] = {"total_detected": total_count, "position_21_count": pos21, "cv_unreliable": True}
            result["analysis_method"] = "CV+VLM (CV-assisted fallback)"
            return result

        cv_text = f"""
COMPUTER VISION ANALYSIS RESULTS:
- Total chromosomes detected: {total_count}
- Position 21 count: {pos21}
- Position 22 count: {pos_counts.get("position_22", 0)}
- Sex chromosome region count: {sex_count}
- Estimated sex chromosomes: {sex_est}
- CV preliminary diagnosis: {ka.get("preliminary_diagnosis", "Unknown")}
- CV confidence: {ka.get("confidence", "low")}
"""
        interp_prompt = CV_VLM_INTERPRETATION_PROMPT.format(
            cv_results=cv_text, total_count=total_count, pos21_count=pos21,
        )
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a clinical cytogeneticist. Interpret the CV counts. Do NOT count visually."},
                {"role": "user", "content": interp_prompt},
            ],
            max_tokens=1500, temperature=0.1,
        )
        raw = response.choices[0].message.content
        st.session_state.raw_response = f"CV Results:\n{cv_text}\n\nVLM Interpretation:\n{raw}"
        result = parse_response(raw, "CV + VLM (Hybrid)")
        if result.get("confidence", 0) < 50:
            st.warning(f"Low confidence ({result.get('confidence', 0)}%). Re-analyzing with visual verification.")
            result = analyze_with_cv_hints(self.api_key, image, pos21, sex_est, sex_count)
            result["fallback_used"] = True
            result["cv_detection"] = {"total_detected": total_count, "position_21_count": pos21, "cv_unreliable": False}
            result["analysis_method"] = "CV+VLM (low-confidence fallback)"
            return result
        result["cv_detection"] = {"total_detected": total_count, "position_21_count": pos21, "sex_chr_estimated": sex_est}
        result["analysis_method"] = "CV+VLM"
        return result

    def _analyze_with_precision_lens(self, image: Image.Image) -> Dict:
        if not self.api_key:
            raise ValueError("API key is required for Precision Clinical Lens")
        provider_map = {APIProvider.OPENAI: "openai", APIProvider.ANTHROPIC: "anthropic", APIProvider.GEMINI: "gemini"}
        vlm_provider = provider_map.get(self.provider)
        if vlm_provider is None:
            if OPENAI_AVAILABLE: vlm_provider = "openai"
            elif ANTHROPIC_AVAILABLE: vlm_provider = "anthropic"
            elif GEMINI_AVAILABLE: vlm_provider = "gemini"
            else: raise ImportError("No VLM API package available.")
        from vlm.precision_lens import PrecisionClinicalLens
        lens = PrecisionClinicalLens(api_key=self.api_key, provider=vlm_provider)
        if "precision_lens_stage" not in st.session_state:
            st.session_state.precision_lens_stage = (0, "Initializing...")
        def cb(num, name): st.session_state.precision_lens_stage = (num, name)
        result = lens.run_pipeline(image, progress_callback=cb)
        st.session_state.raw_response = json.dumps(result.get("stages", {}), indent=2, ensure_ascii=False, default=str)
        return result

    # -- Consensus & mock -----------------------------------------------

    def _analyze_with_consensus(self, image: Image.Image, api_keys: Dict) -> Dict:
        return calculate_consensus(self, image, api_keys)

    def _mock_analysis(self) -> Dict:
        import random
        count = random.choice([46, 47, 45])
        sex = random.choice(["XX", "XY"])
        abnormalities: List = []
        confidence = random.uniform(75, 92)
        if count == 47:
            t = random.choice([21, 18, 13])
            abnormalities.append({"type": "trisomy", "chromosome": str(t), "description": f"Trisomy {t} detected"})
            notation = f"47,{sex},+{t}"
        elif count == 45:
            if sex == "XX":
                abnormalities.append({"type": "monosomy", "chromosome": "X", "description": "Turner syndrome"})
                notation = "45,X"; sex = "X"
            else:
                notation = f"45,{sex},-21"
                abnormalities.append({"type": "monosomy", "chromosome": "21", "description": "Monosomy 21"})
        else:
            notation = f"46,{sex}"
        if random.random() < 0.15 and count == 46:
            abnormalities.append({"type": "translocation", "chromosome": "9;22", "description": "t(9;22)(q34;q11)"})
            notation += ",t(9;22)(q34;q11)"
        return {
            "notation": notation, "chromosome_count": count, "sex_chromosomes": sex,
            "abnormalities": abnormalities, "confidence": round(confidence, 1),
            "interpretation": generate_interpretation(count, abnormalities, sex),
            "detailed_findings": "Demo mode: simulated data.", "analysis_time": datetime.now().isoformat(),
            "technical_notes": "Demo Mode", "provider": "Demo Mode",
        }


def CV2_AVAILABLE_CHECK():
    from providers import CV2_AVAILABLE
    return CV2_AVAILABLE
