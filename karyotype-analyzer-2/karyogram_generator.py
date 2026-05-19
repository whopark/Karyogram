"""
karyogram_generator.py

Public API for rendering classified chromosome crops into a standard
clinical karyogram grid image using PIL.

Rendering helpers live in karyogram_render_helpers.py.
"""

from __future__ import annotations

from PIL import Image

from karyogram_render_helpers import render_row, render_grid

# ---------------------------------------------------------------------------
# Standard Denver group layout definition.
# Each row is a list of position specs; "|" marks a Denver group separator.
# Each position spec is a 1-tuple containing the chromosome label string.
# ---------------------------------------------------------------------------
KARYOGRAM_ROWS: list[list] = [
    [("chr1",), ("chr2",), ("chr3",), "|", ("chr4",), ("chr5",)],
    [("chr6",), ("chr7",), ("chr8",), ("chr9",)],
    [("chr10",), ("chr11",), ("chr12",)],
    [("chr13",), ("chr14",), ("chr15",)],
    [("chr16",), ("chr17",), ("chr18",)],
    [("chr19",), ("chr20",)],
    [("chr21",), ("chr22",), "|", ("chrX",), ("chrY",)],
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_chromosomes(classifications: list[dict]) -> dict[str, list[dict]]:
    """Group classification dicts by chromosome label.

    Returns dict[label -> list[dict]], preserving input order within each group.
    """
    grouped: dict[str, list[dict]] = {}
    for item in classifications:
        label = item.get("label", "")
        grouped.setdefault(label, []).append(item)
    return grouped


def _resolve_crops(
    classifications: list[dict],
    source_image: Image.Image | None,
) -> list[dict]:
    """Ensure every classification has a usable 'crop' PIL.Image.

    Falls back to bbox-based crop from source_image when 'crop' is absent.
    Sets crop=None when neither crop nor source_image is available
    (placeholder rendered by the row renderer).
    """
    resolved = []
    for item in classifications:
        entry = dict(item)
        if entry.get("crop") is None and source_image is not None:
            bbox = entry.get("bbox")
            if bbox and len(bbox) == 4:
                x, y, w, h = (int(v) for v in bbox)
                if w > 0 and h > 0:
                    try:
                        entry["crop"] = source_image.crop((x, y, x + w, y + h))
                    except Exception:
                        entry["crop"] = None
        resolved.append(entry)
    return resolved


def _build_metadata(grouped: dict[str, list[dict]]) -> dict:
    """Build a summary metadata dict from grouped chromosome data."""
    total = sum(len(v) for v in grouped.values())

    trisomies = [lbl for lbl, copies in grouped.items() if len(copies) >= 3]

    # Single copy of an autosome or X is a monosomy; lone Y is normal in males
    monosomies = [
        lbl for lbl, copies in grouped.items()
        if len(copies) == 1 and lbl != "chrY"
    ]

    sex_chroms: list[str] = []
    for lbl in ("chrX", "chrY"):
        sex_chroms.extend([lbl] * len(grouped.get(lbl, [])))

    return {
        "total_chromosomes": total,
        "trisomies": trisomies,
        "monosomies": monosomies,
        "sex_chromosomes": sex_chroms,
        "chromosome_counts": {lbl: len(copies) for lbl, copies in grouped.items()},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_karyogram(
    classifications: list[dict],
    source_image: Image.Image | None = None,
    target_height: int = 100,
) -> tuple[Image.Image, dict]:
    """Render a standard clinical karyogram grid from chromosome classifications.

    Parameters
    ----------
    classifications:
        List of dicts, each containing:
        - "label": str  e.g. "chr1", "chrX", "chrY"
        - "crop": PIL.Image or None  (cropped chromosome image)
        - "confidence": float 0-1
        - "bbox": [x, y, w, h] used for fallback cropping when crop is None
    source_image:
        Original metaphase image used as fallback for bbox-based cropping.
    target_height:
        Uniform pixel height for each chromosome crop in the grid (default 100).

    Returns
    -------
    (karyogram_image, metadata_dict)
        karyogram_image: PIL.Image of the rendered karyogram.
        metadata_dict: summary with keys:
            "total_chromosomes", "trisomies", "monosomies",
            "sex_chromosomes", "chromosome_counts".
    """
    resolved = _resolve_crops(classifications, source_image)
    grouped = _group_chromosomes(resolved)

    row_images = [
        render_row(row_spec, grouped, target_height)
        for row_spec in KARYOGRAM_ROWS
    ]

    karyogram = render_grid(row_images)
    metadata = _build_metadata(grouped)
    return karyogram, metadata
