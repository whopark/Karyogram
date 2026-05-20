"""Precision Clinical Lens: 6-stage sequential VLM pipeline for
karyotype analysis."""

import io
import json
import re
import base64
from datetime import datetime
from typing import Dict, Optional
from PIL import Image

from providers import (
    OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE, GEMINI_AVAILABLE,
)

if OPENAI_AVAILABLE:
    from openai import OpenAI
if ANTHROPIC_AVAILABLE:
    import anthropic
if GEMINI_AVAILABLE:
    from google import genai

from vlm.precision_lens_stages import (
    stage_counting,
    stage_classification,
    stage_cluster_classification,
    stage_translocation,
    stage_analysis,
    stage_abnormality_detection,
)


class PrecisionClinicalLens:
    """6-stage sequential VLM pipeline for precision karyotype analysis."""

    STAGE_NAMES = [
        ("counting", "Counting"),
        ("classification", "Classification"),
        ("cluster_classification", "Cluster Classification"),
        ("translocation", "Translocation Detection"),
        ("analysis", "Comprehensive Analysis"),
        ("abnormality_detection", "Abnormality Detection"),
    ]

    def __init__(self, api_key: str, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider

    def _encode_image_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    def _call_vlm(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        """Call VLM API and return raw text response."""
        image_base64 = self._encode_image_base64(image)

        if self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                        {"type": "text", "text": user_prompt},
                    ],
                }],
            )
            return response.content[0].text

        if self.provider == "gemini" and GEMINI_AVAILABLE:
            client = genai.Client(api_key=self.api_key)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[system_prompt + "\n\n" + user_prompt, image],
                config={"temperature": 0.1, "max_output_tokens": 2500},
            )
            return response.text

        # Default: OpenAI
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}},
                ]},
            ],
            max_tokens=2500,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _parse_json(self, raw: str) -> Dict:
        """Extract JSON from VLM response."""
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = re.search(pattern, raw)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw_text": raw, "parse_error": True}

    def run_pipeline(self, image: Image.Image, progress_callback=None) -> Dict:
        """Run the full 6-stage pipeline."""
        from providers import CV2_AVAILABLE
        stages: Dict = {}
        cv_data = None
        if CV2_AVAILABLE:
            from cv.detector import ChromosomeDetector
            detector = ChromosomeDetector()
            cv_data = detector.detect_chromosomes(image)

        call = self._call_vlm
        parse = self._parse_json

        if progress_callback:
            progress_callback(1, "Counting")
        stages["stage_1"] = stage_counting(call, parse, image, cv_data)

        if progress_callback:
            progress_callback(2, "Classification")
        stages["stage_2"] = stage_classification(call, parse, image, stages["stage_1"])

        if progress_callback:
            progress_callback(3, "Cluster Classification")
        stages["stage_3"] = stage_cluster_classification(call, parse, image, stages["stage_1"], stages["stage_2"])

        if progress_callback:
            progress_callback(4, "Translocation Detection")
        stages["stage_4"] = stage_translocation(call, parse, image, stages["stage_3"])

        if progress_callback:
            progress_callback(5, "Comprehensive Analysis")
        stages["stage_5"] = stage_analysis(call, parse, image, stages)

        if progress_callback:
            progress_callback(6, "Abnormality Detection")
        stages["stage_6"] = stage_abnormality_detection(call, parse, image, stages)

        final = stages["stage_6"]
        return {
            "notation": final.get("notation", "Unable to determine"),
            "chromosome_count": final.get("chromosome_count", stages["stage_1"].get("total_count", 0)),
            "sex_chromosomes": final.get("sex_chromosomes", "Unknown"),
            "abnormalities": final.get("abnormalities", []),
            "confidence": final.get("confidence", 0),
            "interpretation": final.get("interpretation", ""),
            "detailed_findings": final.get("detailed_findings", ""),
            "analysis_time": datetime.now().isoformat(),
            "technical_notes": "Precision Clinical Lens - 6-Stage Sequential Pipeline",
            "provider": f"Precision Clinical Lens ({self.provider})",
            "pipeline": "precision_lens",
            "stages": stages,
            "cv_data": cv_data,
        }
