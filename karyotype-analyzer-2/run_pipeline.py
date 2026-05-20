"""Batch Precision Clinical Lens pipeline for metaphase spread images.

Runs the full 6-stage pipeline:
  1. Counting - chromosome detection and counting
  2. Classification - Denver group classification
  3. Cluster Classification - individual chromosome ID & pairing
  4. Translocation Detection - structural abnormality scan
  5. Comprehensive Analysis - cross-validation & synthesis
  6. Abnormality Detection - final ISCN diagnosis
"""

import os
import sys
import json
import time
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
from providers import CV2_AVAILABLE
from vlm import PrecisionClinicalLens
from cv import ChromosomeDetector


def run_pipeline(image_dir: str, provider: str = "openai"):
    """Run 6-stage Precision Clinical Lens on all images."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} not found")
        return

    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    if provider not in key_map:
        print(f"Available providers: {', '.join(key_map.keys())}")
        return

    api_key = os.environ.get(key_map[provider], "")
    if not api_key:
        print(f"Error: {key_map[provider]} not set")
        return

    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in extensions
    )
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"{'=' * 70}")
    print(f"  Precision Clinical Lens - 6-Stage Pipeline")
    print(f"  Provider : {provider}")
    print(f"  CV engine: {'OpenCV available' if CV2_AVAILABLE else 'not available (VLM-only)'}")
    print(f"  Images   : {len(images)} files from {image_dir}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    lens = PrecisionClinicalLens(api_key=api_key, provider=provider)
    results = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        start = time.time()

        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            def progress(stage_num, stage_name):
                print(f"  Stage {stage_num}/6: {stage_name}...", flush=True)

            result = lens.run_pipeline(image, progress_callback=progress)
            elapsed = time.time() - start

            notation = result.get('notation', 'N/A')
            count = result.get('chromosome_count', '?')
            sex = result.get('sex_chromosomes', '?')
            conf = result.get('confidence', '?')
            abnormalities = result.get('abnormalities', [])

            print(f"  => {notation}  (n={count}, {sex}, conf={conf}%)", end="")
            if abnormalities:
                types = [a.get('type', a.get('subtype', '')) for a in abnormalities]
                print(f"  [ABNORMAL: {', '.join(types)}]", end="")
            print(f"  [{elapsed:.1f}s]\n")

            result['_source_file'] = img_path.name
            result['_elapsed_seconds'] = round(elapsed, 1)
            results.append(result)

        except Exception as e:
            elapsed = time.time() - start
            print(f"  => ERROR: {e}  [{elapsed:.1f}s]\n")
            results.append({
                '_source_file': img_path.name,
                '_elapsed_seconds': round(elapsed, 1),
                'error': str(e),
            })

    # ── Summary ──
    print(f"{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    for r in successful:
        notation = r.get('notation', 'N/A')
        count = r.get('chromosome_count', '?')
        conf = r.get('confidence', '?')
        abnormal = " *" if r.get('abnormalities') else ""
        print(f"  {r['_source_file']:25s}  {notation:25s}  n={count}  conf={conf}%{abnormal}")

    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for r in failed:
            print(f"  {r['_source_file']:25s}  {r['error']}")

    normal = sum(1 for r in successful if not r.get('abnormalities'))
    abnormal = sum(1 for r in successful if r.get('abnormalities'))
    total_time = sum(r.get('_elapsed_seconds', 0) for r in results)
    print(f"\n  Normal: {normal}  |  Abnormal: {abnormal}  |  Failed: {len(failed)}")
    print(f"  Total time: {total_time:.0f}s  |  Avg: {total_time/len(results):.1f}s/image")

    output_path = image_dir / "precision_lens_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Precision Clinical Lens — 6-stage karyotype analysis"
    )
    parser.add_argument("image_dir", help="Directory with metaphase images")
    parser.add_argument(
        "--provider", default="openai",
        choices=["openai", "anthropic", "gemini"],
        help="VLM provider (default: openai)",
    )
    args = parser.parse_args()
    run_pipeline(args.image_dir, args.provider)
