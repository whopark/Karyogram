"""Pair refinement for chromosome classification post-processing.

Handles autosome pairing (each autosome=2) and sex chromosome assignment
by elimination + crop size discrimination.
"""

IDX_TO_LABEL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# @AX:NOTE: [AUTO] two-phase refinement — Phase 1: autosome pairing (each=2), Phase 2: sex chromosome assignment by elimination + crop size
def pair_refine(labels: list, confs: list, crop_areas: list = None,
                max_iter: int = 10) -> list:
    """Reassign autosomes to pairs, then assign sex chromosomes by elimination.

    Phase 1: Over-represented autosomes (target=2) are reassigned to under-represented ones
    using lowest-confidence swap. Phase 2: After autosome pairing, remaining excess chromosomes
    in Group C (chr6-12/chrX) and Group G (chr21-22/chrY) are reassigned as sex chromosomes
    using crop area to distinguish chrX (larger) from chrY (smaller).
    """
    labels, confs = list(labels), list(confs)
    autosome_labels = IDX_TO_LABEL[:22]

    # Phase 1: Autosome pairing
    for _ in range(max_iter):
        counts = {lbl: labels.count(lbl) for lbl in IDX_TO_LABEL}
        changed = False
        for lbl in autosome_labels:
            while counts.get(lbl, 0) > 2:
                under = min(
                    (l for l in autosome_labels if counts.get(l, 0) < 2),
                    key=lambda l: counts.get(l, 0),
                    default=None,
                )
                if under is None:
                    break
                lbl_indices = [i for i, l in enumerate(labels) if l == lbl]
                worst = min(lbl_indices, key=lambda i: confs[i])
                labels[worst] = under
                counts[lbl] -= 1
                counts[under] = counts.get(under, 0) + 1
                changed = True
        if not changed:
            break

    # Phase 2: Sex chromosome assignment by elimination + size
    counts = {lbl: labels.count(lbl) for lbl in IDX_TO_LABEL}
    x_count = counts.get("chrX", 0)
    y_count = counts.get("chrY", 0)
    sex_total = x_count + y_count

    if sex_total < 2:
        needed = 2 - sex_total
        group_c = {"chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12"}
        group_g = {"chr21", "chr22"}
        candidates = []
        for i, lbl in enumerate(labels):
            if lbl in group_c or lbl in group_g:
                area = crop_areas[i] if crop_areas and i < len(crop_areas) else 0
                candidates.append((i, lbl, confs[i], area))
        candidates.sort(key=lambda x: x[2])  # lowest confidence first
        reassigned = [(idx, area) for idx, lbl, conf, area in candidates[:needed]]

        if len(reassigned) == 2 and crop_areas:
            reassigned.sort(key=lambda x: x[1], reverse=True)  # larger = chrX
            labels[reassigned[0][0]] = "chrX"
            labels[reassigned[1][0]] = "chrY"
        elif len(reassigned) == 2:
            for idx, _ in reassigned:
                labels[idx] = "chrX" if labels[idx] in group_c else "chrY"
        elif len(reassigned) == 1:
            idx = reassigned[0][0]
            if x_count == 0:
                labels[idx] = "chrX"
            elif y_count == 0:
                labels[idx] = "chrY"
            else:
                labels[idx] = "chrX"

    return labels
