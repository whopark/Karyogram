"""Run ALL analysis modes on metaphase images and produce comparison table.

Modes:
  1. Single VLM (metaphase-specific prompt)
  2. Two-Stage (CV detection -> VLM classification)
  3. CV+VLM Hybrid (CV counting -> VLM interpretation)
  4. Precision Clinical Lens (6-stage VLM pipeline)
"""

import os
import sys
import json
import time
import base64
import io
import re
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

from app import (
    KaryotypeAnalyzer, APIProvider, PrecisionClinicalLens,
    ChromosomeDetector, CV2_AVAILABLE
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


def extract_result(r: dict) -> dict:
    return {
        "notation": r.get("notation", "ERROR"),
        "count": r.get("chromosome_count", r.get("total_chromosome_count", "?")),
        "sex": r.get("sex_chromosomes", "?"),
        "confidence": r.get("confidence", "?"),
        "abnormal": bool(r.get("abnormalities")),
        "abnormalities": r.get("abnormalities", []),
    }


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

    # ── Comparison Table ──
    print(f"\n{'=' * 150}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 150}")

    header = f"{'File':25s}"
    for m_name, _ in METHODS:
        header += f" | {m_name:30s}"
    header += " | Consensus"
    print(header)
    print("-" * 150)

    for fname in sorted(all_results.keys()):
        methods = all_results[fname]
        row = f"{fname:25s}"
        notations = []

        for m_name, _ in METHODS:
            r = methods.get(m_name, {})
            if "error" in r:
                cell = "ERROR"
            else:
                ex = extract_result(r)
                nota = ex["notation"]
                if len(nota) > 25:
                    nota = nota[:22] + "..."
                cell = f"{nota} (n={ex['count']})"
                notations.append(ex["notation"].split(" or ")[0].strip())
            row += f" | {cell:30s}"

        # Simple consensus: most common notation
        if notations:
            from collections import Counter
            counts = Counter(notations)
            top, top_count = counts.most_common(1)[0]
            agreement = f"{top_count}/{len(notations)}"
            row += f" | {top} [{agreement}]"
        else:
            row += f" | N/A"

        print(row)

    print("-" * 150)

    # ── Statistics ──
    print(f"\n{'=' * 150}")
    print(f"  STATISTICS BY METHOD")
    print(f"{'=' * 150}")

    for m_name, _ in METHODS:
        results = [all_results[f].get(m_name, {}) for f in sorted(all_results)]
        ok = [r for r in results if "error" not in r]
        errs = len(results) - len(ok)

        extracted = [extract_result(r) for r in ok]
        normal = sum(1 for e in extracted if not e["abnormal"])
        abnormal = sum(1 for e in extracted if e["abnormal"])

        counts = [e["count"] for e in extracted if isinstance(e["count"], int)]
        avg_count = sum(counts) / len(counts) if counts else 0

        confs = [e["confidence"] for e in extracted if isinstance(e["confidence"], (int, float))]
        avg_conf = sum(confs) / len(confs) if confs else 0

        times = [all_results[f].get(m_name, {}).get("_elapsed", 0) for f in all_results]
        total_time = sum(times)

        nota_dist = {}
        for e in extracted:
            n = e["notation"].split(" or ")[0].strip()
            # Simplify to base notation
            if "46," in n:
                key = "46,N (normal)"
            elif "47," in n and "+21" in n:
                key = "47,+21 (T21)"
            elif "45," in n:
                key = "45,X (Turner)"
            elif "92," in n:
                key = "92 (tetraploid)"
            elif "55," in n:
                key = "55+ (hyperdiploid)"
            else:
                key = n[:20]
            nota_dist[key] = nota_dist.get(key, 0) + 1

        print(f"\n  {m_name}")
        print(f"    Normal: {normal}  Abnormal: {abnormal}  Errors: {errs}")
        print(f"    Avg count: {avg_count:.1f}  Avg confidence: {avg_conf:.1f}%")
        print(f"    Total time: {total_time:.0f}s  Avg: {total_time/len(results):.1f}s/image")
        print(f"    Distribution: {nota_dist}")

    # ── Agreement Matrix ──
    print(f"\n{'=' * 150}")
    print(f"  INTER-METHOD AGREEMENT MATRIX")
    print(f"{'=' * 150}")

    method_names = [m[0] for m in METHODS]
    files = sorted(all_results.keys())

    # Build notation vectors per method
    nota_vectors = {}
    for m_name in method_names:
        vec = []
        for f in files:
            r = all_results[f].get(m_name, {})
            if "error" in r:
                vec.append("ERROR")
            else:
                vec.append(extract_result(r)["notation"].split(" or ")[0].strip())
        nota_vectors[m_name] = vec

    # Pairwise agreement
    print(f"\n  {'':20s}", end="")
    for m in method_names:
        print(f" {m:15s}", end="")
    print()

    for m1 in method_names:
        print(f"  {m1:20s}", end="")
        for m2 in method_names:
            if m1 == m2:
                print(f" {'--':>15s}", end="")
            else:
                agree = sum(1 for a, b in zip(nota_vectors[m1], nota_vectors[m2]) if a == b and a != "ERROR")
                total = sum(1 for a, b in zip(nota_vectors[m1], nota_vectors[m2]) if a != "ERROR" and b != "ERROR")
                pct = (agree / total * 100) if total > 0 else 0
                print(f" {f'{agree}/{total} ({pct:.0f}%)':>15s}", end="")
        print()

    # Save all results
    output_path = image_dir / "all_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  All results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all karyotype analysis methods")
    parser.add_argument("image_dir", help="Directory with metaphase images")
    args = parser.parse_args()
    run_all(args.image_dir)
