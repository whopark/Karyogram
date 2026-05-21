"""Run ALL analysis modes on metaphase images and produce comparison table.

Modes:
  1. Single VLM (metaphase-specific prompt)
  2. Two-Stage (CV detection -> VLM classification)
  3. CV+VLM Hybrid (CV counting -> VLM interpretation)
  4. Precision Clinical Lens (6-stage VLM pipeline)
"""

import os
import sys
import time
import base64
import io
import re
import json
from pathlib import Path
from datetime import datetime
from PIL import Image

# Stub streamlit for headless usage
_fake_state = {}

class _FakeSessionState:
    def __getattr__(self, name):
        return _fake_state.get(name)
    def __setattr__(self, name, value):
        _fake_state[name] = value
    def __contains__(self, name):
        return name in _fake_state
    def __getitem__(self, name):
        return _fake_state.get(name)
    def __setitem__(self, name, value):
        _fake_state[name] = value

class _FakeStreamlit:
    session_state = _FakeSessionState()
    def __getattr__(self, name):
        return lambda *a, **kw: None

sys.modules['streamlit'] = _FakeStreamlit()
sys.path.insert(0, str(Path(__file__).parent))

from providers import APIProvider, CV2_AVAILABLE
from vlm import KaryotypeAnalyzer, PrecisionClinicalLens
from cv import ChromosomeDetector
from run_all_report import (
    extract_result, print_comparison_table, print_statistics,
    print_agreement_matrix, save_results,
)

# ── Metaphase-specific single VLM prompt ──

METAPHASE_PROMPT = """You are an expert clinical cytogeneticist analyzing a METAPHASE SPREAD image.

This image shows chromosomes scattered from a single cell during metaphase.
The chromosomes are NOT arranged or labeled.

## TASK
1. Count all visible chromosomes carefully
2. Classify into Denver groups by size/morphology
3. Identify sex chromosomes if possible
4. Detect numerical abnormalities

## COUNTING GUIDELINES
- Count each distinct chromosome (X-shaped or rod-shaped dark structures)
- Ignore interphase nuclei (large dark round blobs) and debris
- Normal human cell: 46 chromosomes

## OUTPUT (JSON only)
{
    "notation": "ISCN notation",
    "chromosome_count": number,
    "sex_chromosomes": "XX/XY/XXY/XXX/X/unknown",
    "denver_group_counts": {
        "A_1_3": n, "B_4_5": n, "C_6_12_X": n,
        "D_13_15": n, "E_16_18": n, "F_19_20": n, "G_21_22_Y": n
    },
    "abnormalities": [{"type":"","chromosome":"","description":""}],
    "confidence": number,
    "interpretation": "clinical interpretation",
    "detailed_findings": "description"
}"""


def encode_image(img: Image.Image) -> str:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def parse_json(raw: str) -> dict:
    if not raw or not raw.strip():
        return {"error": "Empty response"}
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    for pat in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
        m = re.search(pat, raw)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    start = raw.find('{')
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(raw)):
            c = raw[i]
            if esc:
                esc = False
                continue
            if c == '\\' and in_str:
                esc = True
                continue
            if c == '"' and not esc:
                in_str = not in_str
                continue
            if not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw[start:i + 1])
                        except json.JSONDecodeError:
                            break
    return {"error": "parse_failed", "raw_preview": raw[:200]}


# ── Analysis functions ──

def analyze_single_vlm(img: Image.Image, api_key: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    b64 = encode_image(img)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a board-certified clinical cytogeneticist analyzing a RAW METAPHASE SPREAD. Count carefully. Return valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": METAPHASE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
            ]}
        ],
        max_tokens=2500, temperature=0.1,
    )
    return parse_json(resp.choices[0].message.content)


def analyze_two_stage(img: Image.Image, api_key: str) -> dict:
    analyzer = KaryotypeAnalyzer(provider=APIProvider.TWO_STAGE, api_key=api_key)
    return analyzer.analyze(img)


def analyze_cv_vlm(img: Image.Image, api_key: str) -> dict:
    analyzer = KaryotypeAnalyzer(provider=APIProvider.CV_VLM, api_key=api_key)
    return analyzer.analyze(img)


def analyze_precision_lens(img: Image.Image, api_key: str) -> dict:
    lens = PrecisionClinicalLens(api_key=api_key, provider="openai")
    return lens.run_pipeline(img)


# ── Main pipeline ──

METHODS = [
    ("Single VLM", analyze_single_vlm),
    ("Two-Stage", analyze_two_stage),
    ("CV+VLM", analyze_cv_vlm),
    ("Precision Lens", analyze_precision_lens),
]


def run_all(image_dir: str):
    image_dir = Path(image_dir)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in extensions)

    print(f"{'=' * 70}")
    print(f"  Comprehensive Karyotype Analysis - All Methods")
    print(f"  CV engine: {'OpenCV available' if CV2_AVAILABLE else 'NOT available'}")
    print(f"  Images:    {len(images)} files")
    print(f"  Methods:   {', '.join(m[0] for m in METHODS)}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    all_results = {}  # {filename: {method_name: result}}

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        all_results[img_path.name] = {}

        for method_name, method_fn in METHODS:
            print(f"  {method_name:20s}...", end=" ", flush=True)
            start = time.time()
            try:
                result = method_fn(image, api_key)
                elapsed = time.time() - start
                ex = extract_result(result)
                flag = " [ABN]" if ex["abnormal"] else ""
                print(f"{ex['notation']:30s} n={ex['count']}  {ex['sex']:6s}  conf={ex['confidence']}%{flag}  [{elapsed:.1f}s]")
                result['_elapsed'] = round(elapsed, 1)
                all_results[img_path.name][method_name] = result
            except Exception as e:
                elapsed = time.time() - start
                print(f"ERROR: {e}  [{elapsed:.1f}s]")
                all_results[img_path.name][method_name] = {"error": str(e), "_elapsed": round(elapsed, 1)}
        print()

    method_names = [m[0] for m in METHODS]
    print_comparison_table(all_results, method_names)
    print_statistics(all_results, method_names)
    print_agreement_matrix(all_results, method_names)
    save_results(all_results, image_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all karyotype analysis methods")
    parser.add_argument("image_dir", help="Directory with metaphase images")
    args = parser.parse_args()
    run_all(args.image_dir)
