"""Reporting utilities for run_all_analyses.py.

Prints comparison tables, per-method statistics, and inter-method
agreement matrices from collected analysis results.
"""

import json
from collections import Counter
from pathlib import Path


def extract_result(r: dict) -> dict:
    """Extract standardized fields from a raw analysis result."""
    return {
        "notation": r.get("notation", "ERROR"),
        "count": r.get("chromosome_count", r.get("total_chromosome_count", "?")),
        "sex": r.get("sex_chromosomes", "?"),
        "confidence": r.get("confidence", "?"),
        "abnormal": bool(r.get("abnormalities")),
        "abnormalities": r.get("abnormalities", []),
    }


def print_comparison_table(all_results: dict, method_names: list) -> None:
    """Print side-by-side comparison of results across methods."""
    print(f"\n{'=' * 150}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 150}")

    header = f"{'File':25s}"
    for m_name in method_names:
        header += f" | {m_name:30s}"
    header += " | Consensus"
    print(header)
    print("-" * 150)

    for fname in sorted(all_results.keys()):
        methods = all_results[fname]
        row = f"{fname:25s}"
        notations = []

        for m_name in method_names:
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

        if notations:
            counts = Counter(notations)
            top, top_count = counts.most_common(1)[0]
            agreement = f"{top_count}/{len(notations)}"
            row += f" | {top} [{agreement}]"
        else:
            row += f" | N/A"

        print(row)

    print("-" * 150)


def print_statistics(all_results: dict, method_names: list) -> None:
    """Print per-method statistics (normal/abnormal/error counts, averages)."""
    print(f"\n{'=' * 150}")
    print(f"  STATISTICS BY METHOD")
    print(f"{'=' * 150}")

    for m_name in method_names:
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

        nota_dist = _classify_notations(extracted)

        print(f"\n  {m_name}")
        print(f"    Normal: {normal}  Abnormal: {abnormal}  Errors: {errs}")
        print(f"    Avg count: {avg_count:.1f}  Avg confidence: {avg_conf:.1f}%")
        print(f"    Total time: {total_time:.0f}s  Avg: {total_time/len(results):.1f}s/image")
        print(f"    Distribution: {nota_dist}")


def _classify_notations(extracted: list) -> dict:
    """Bucket notation strings into simplified categories."""
    nota_dist: dict = {}
    for e in extracted:
        n = e["notation"].split(" or ")[0].strip()
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
    return nota_dist


def print_agreement_matrix(all_results: dict, method_names: list) -> None:
    """Print pairwise inter-method agreement percentages."""
    print(f"\n{'=' * 150}")
    print(f"  INTER-METHOD AGREEMENT MATRIX")
    print(f"{'=' * 150}")

    files = sorted(all_results.keys())
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


def save_results(all_results: dict, image_dir: Path) -> None:
    """Save all results as JSON to the image directory."""
    output_path = image_dir / "all_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  All results saved to: {output_path}")
