"""Precision Clinical Lens stage implementations (1-6)."""

import json
from typing import Callable, Dict, Optional
from PIL import Image


def stage_counting(call_vlm: Callable, parse_json: Callable,
                   image: Image.Image, cv_data: Optional[Dict]) -> Dict:
    cv_hint = ""
    if cv_data and cv_data.get("count", 0) > 0:
        cv_hint = (f"\n\nCV pre-analysis detected approximately {cv_data['count']} chromosomes "
                   f"using method '{cv_data.get('detection_method', 'unknown')}'. "
                   "Use this as a reference but verify visually.")
    raw = call_vlm(
        system_prompt="You are a board-certified clinical cytogeneticist. Your ONLY task in this step is to COUNT chromosomes accurately. Do not diagnose yet.",
        user_prompt=f"""## STAGE 1: CHROMOSOME COUNTING
Your sole task is to count the total number of chromosomes in this karyogram image.
{cv_hint}
### Instructions:
1. Count chromosomes at each labeled position (1-22, X, Y)
2. Record how many chromosomes appear at each position
3. Calculate the total
### Output JSON:
{{
    "total_count": <number>,
    "position_counts": {{"1": 2, "2": 2, ..., "22": 2, "X": <number>, "Y": <number>}},
    "count_notes": "any observations about counting difficulty or ambiguity"
}}""",
        image=image,
    )
    return parse_json(raw)


def stage_classification(call_vlm: Callable, parse_json: Callable,
                         image: Image.Image, stage1: Dict) -> Dict:
    total = stage1.get("total_count", "unknown")
    raw = call_vlm(
        system_prompt="You are a clinical cytogeneticist. Classify chromosomes into Denver groups based on size and centromere position.",
        user_prompt=f"""## STAGE 2: CHROMOSOME CLASSIFICATION
Previous stage counted {total} chromosomes.
### Instructions:
Classify each chromosome into Denver groups:
- **Group A (1-3)**: Large metacentric/submetacentric
- **Group B (4-5)**: Large submetacentric
- **Group C (6-12, X)**: Medium submetacentric
- **Group D (13-15)**: Medium acrocentric
- **Group E (16-18)**: Small metacentric/submetacentric
- **Group F (19-20)**: Small metacentric
- **Group G (21-22, Y)**: Small acrocentric
### Output JSON:
{{
    "denver_groups": {{
        "A": {{"count": <n>, "expected": 6, "chromosomes": "1-3"}},
        "B": {{"count": <n>, "expected": 4, "chromosomes": "4-5"}},
        "C": {{"count": <n>, "expected": 15, "chromosomes": "6-12,X (female) or 14 for male"}},
        "D": {{"count": <n>, "expected": 6, "chromosomes": "13-15"}},
        "E": {{"count": <n>, "expected": 6, "chromosomes": "16-18"}},
        "F": {{"count": <n>, "expected": 4, "chromosomes": "19-20"}},
        "G": {{"count": <n>, "expected": 5, "chromosomes": "21-22,Y (male) or 4 for female"}}
    }},
    "total_classified": <number>,
    "classification_notes": "observations about banding patterns, staining quality"
}}""",
        image=image,
    )
    return parse_json(raw)


def stage_cluster_classification(call_vlm: Callable, parse_json: Callable,
                                 image: Image.Image, stage1: Dict, stage2: Dict) -> Dict:
    pos_counts = json.dumps(stage1.get("position_counts", {}), indent=2)
    denver = json.dumps(stage2.get("denver_groups", {}), indent=2)
    raw = call_vlm(
        system_prompt="You are a clinical cytogeneticist performing detailed chromosome identification. Match each chromosome to its specific number.",
        user_prompt=f"""## STAGE 3: CLUSTER CLASSIFICATION
### Previous Results:
Position counts: {pos_counts}
Denver groups: {denver}
### Instructions:
For each chromosome position (1-22, X, Y), confirm:
1. The number of chromosomes present (normally 2 per autosome)
2. Whether paired chromosomes show normal homolog matching
3. Whether banding patterns are consistent with the labeled position
4. Identify the sex chromosome configuration
### Output JSON:
{{
    "chromosome_pairs": {{
        "1": {{"count": 2, "status": "normal/abnormal", "note": ""}}, ...
        "22": {{"count": 2, "status": "normal", "note": ""}},
        "X": {{"count": <n>, "note": ""}},
        "Y": {{"count": <n>, "note": ""}}
    }},
    "sex_determination": "XX/XY/XXY/XXX/X/XYY",
    "anomalous_positions": ["list of positions with abnormal counts or morphology"],
    "cluster_notes": "observations about homolog pairing and banding consistency"
}}""",
        image=image,
    )
    return parse_json(raw)


def stage_translocation(call_vlm: Callable, parse_json: Callable,
                        image: Image.Image, stage3: Dict) -> Dict:
    anomalous = json.dumps(stage3.get("anomalous_positions", []))
    raw = call_vlm(
        system_prompt="You are a clinical cytogeneticist specializing in structural chromosome abnormalities. Focus on detecting translocations, inversions, deletions, and duplications.",
        user_prompt=f"""## STAGE 4: TRANSLOCATION & STRUCTURAL ABNORMALITY DETECTION
### Previous Results:
Anomalous positions flagged: {anomalous}
### Instructions:
Examine each chromosome pair for structural abnormalities:
1. Translocations: t(A;B)(breakpoints)
2. Inversions: inv(chromosome)(p;q)
3. Deletions: del(chromosome)(breakpoint)
4. Duplications: dup(chromosome)(region)
5. Ring chromosomes: r(chromosome)
6. Isochromosomes: i(chromosome)(arm)
### Output JSON:
{{
    "structural_abnormalities": [
        {{"type": "translocation/inversion/deletion/duplication/ring/isochromosome",
          "iscn_notation": "e.g., t(9;22)(q34;q11.2)",
          "chromosomes_involved": ["9", "22"],
          "description": "detailed description",
          "confidence": <0-100>}}
    ],
    "normal_structure": true/false,
    "translocation_notes": "observations about structural integrity"
}}""",
        image=image,
    )
    return parse_json(raw)


def stage_analysis(call_vlm: Callable, parse_json: Callable,
                   image: Image.Image, all_stages: Dict) -> Dict:
    summary = {
        "total_count": all_stages["stage_1"].get("total_count", "unknown"),
        "sex_determination": all_stages["stage_3"].get("sex_determination", "unknown"),
        "anomalous_positions": all_stages["stage_3"].get("anomalous_positions", []),
        "structural_abnormalities": all_stages["stage_4"].get("structural_abnormalities", []),
        "normal_structure": all_stages["stage_4"].get("normal_structure", True),
    }
    raw = call_vlm(
        system_prompt="You are a senior clinical cytogeneticist performing comprehensive karyotype analysis. Synthesize all previous findings.",
        user_prompt=f"""## STAGE 5: COMPREHENSIVE ANALYSIS
### Accumulated Findings:
{json.dumps(summary, indent=2)}
### Instructions:
1. Numerical Analysis: Confirm total count, identify aneuploidy
2. Structural Analysis: Review detected structural changes
3. Sex Chromosome Analysis: Confirm sex chromosome constitution
4. Cross-validation: Check consistency between stages
### Output JSON:
{{
    "numerical_status": "normal/aneuploidy type",
    "structural_status": "normal/abnormal with details",
    "sex_chromosome_status": "normal/abnormal with details",
    "cross_validation": {{"count_consistent": true/false, "groups_consistent": true/false, "discrepancies": []}},
    "preliminary_karyotype": "ISCN notation",
    "analysis_summary": "comprehensive text summary"
}}""",
        image=image,
    )
    return parse_json(raw)


def stage_abnormality_detection(call_vlm: Callable, parse_json: Callable,
                                image: Image.Image, all_stages: Dict) -> Dict:
    s1 = all_stages["stage_1"]
    s3 = all_stages["stage_3"]
    s4 = all_stages["stage_4"]
    s5 = all_stages["stage_5"]
    findings = {
        "total_count": s1.get("total_count", "unknown"),
        "sex_determination": s3.get("sex_determination", "unknown"),
        "anomalous_positions": s3.get("anomalous_positions", []),
        "structural_abnormalities": s4.get("structural_abnormalities", []),
        "preliminary_karyotype": s5.get("preliminary_karyotype", "unknown"),
        "numerical_status": s5.get("numerical_status", "unknown"),
        "structural_status": s5.get("structural_status", "unknown"),
        "cross_validation": s5.get("cross_validation", {}),
    }
    raw = call_vlm(
        system_prompt="You are a chief cytogeneticist issuing the FINAL clinical karyotype report. Produce the definitive ISCN 2020 notation and clinical interpretation.",
        user_prompt=f"""## STAGE 6: FINAL ABNORMALITY DETECTION & DIAGNOSIS
### All Pipeline Findings:
{json.dumps(findings, indent=2)}
### Instructions:
1. Final ISCN Notation: Complete, ISCN 2020-compliant
2. Abnormality Classification: numerical, structural, combined
3. Clinical Correlation: known syndromes, prognostic implications
4. Confidence Assessment
### Output JSON:
{{
    "notation": "Complete ISCN 2020 notation",
    "chromosome_count": <number>,
    "sex_chromosomes": "XX/XY/X/XXY/XXX/XYY",
    "abnormalities": [
        {{"type": "numerical/structural", "subtype": "trisomy/monosomy/translocation/etc",
          "chromosome": "affected", "iscn_detail": "sub-notation",
          "description": "detailed", "clinical_significance": "associated syndrome"}}
    ],
    "confidence": <0-100>,
    "interpretation": "Complete clinical interpretation paragraph",
    "detailed_findings": "Summary of all 6 stages",
    "recommendations": "suggested follow-up if abnormalities found"
}}""",
        image=image,
    )
    return parse_json(raw)
