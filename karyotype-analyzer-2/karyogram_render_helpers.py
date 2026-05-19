"""
karyogram_render_helpers.py

Low-level PIL rendering helpers for the karyogram generator.
Handles font loading, placeholder images, pair composition, and row assembly.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Color and spacing constants
# ---------------------------------------------------------------------------
BG_COLOR = (255, 255, 255)
LABEL_COLOR = (30, 30, 30)
GROUP_LABEL_COLOR = (80, 80, 180)
SEPARATOR_COLOR = (180, 180, 180)
PLACEHOLDER_COLOR = (210, 210, 210)
TRISOMY_BORDER_COLOR = (220, 30, 30)
MONOSOMY_DASH_COLOR = (150, 150, 150)

PAIR_GAP = 8          # px between two homologs
POSITION_GAP = 25     # px between chromosome positions
SEPARATOR_GAP = 40    # px for Denver group separator "|"
LABEL_HEIGHT = 28     # px reserved below each pair for the label
ROW_VGAP = 15         # px vertical gap between rows
MARGIN = 20           # px outer margin
TRISOMY_BORDER = 2    # px red border width for trisomy extra copy

# Denver group name mapped to the first chromosome label in that segment
DENVER_GROUP_LABELS: dict[str, str] = {
    "chr1": "A", "chr4": "B",
    "chr6": "C",
    "chr10": "C",
    "chr13": "D",
    "chr16": "E",
    "chr19": "F",
    "chr21": "G", "chrX": "sex",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Attempt to load a system TrueType font; fall back to PIL default."""
    candidates = [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "FreeSansBold.ttf" if bold else "FreeSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def normalize_crop(crop_image: Image.Image, target_height: int) -> Image.Image:
    """Resize crop to target_height preserving aspect ratio; convert to grayscale RGB."""
    w, h = crop_image.size
    if h == 0:
        return Image.new("RGB", (target_height // 2, target_height), color=BG_COLOR)
    scale = target_height / h
    new_w = max(1, int(w * scale))
    resized = crop_image.resize((new_w, target_height), Image.LANCZOS)
    return resized.convert("L").convert("RGB")


def placeholder_image(width: int, height: int, dashed: bool = False) -> Image.Image:
    """Create a placeholder rectangle for a missing homolog.

    When dashed=True, draws a dashed border to indicate monosomy.
    """
    img = Image.new("RGB", (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    if dashed:
        dash_len, gap_len = 6, 4
        x0, y0, x1, y1 = 1, 1, width - 2, height - 2
        for axis in ("h_top", "h_bot", "v_left", "v_right"):
            if axis == "h_top":
                x = x0
                while x < x1:
                    draw.line([(x, y0), (min(x + dash_len, x1), y0)], fill=MONOSOMY_DASH_COLOR)
                    x += dash_len + gap_len
            elif axis == "h_bot":
                x = x0
                while x < x1:
                    draw.line([(x, y1), (min(x + dash_len, x1), y1)], fill=MONOSOMY_DASH_COLOR)
                    x += dash_len + gap_len
            elif axis == "v_left":
                y = y0
                while y < y1:
                    draw.line([(x0, y), (x0, min(y + dash_len, y1))], fill=MONOSOMY_DASH_COLOR)
                    y += dash_len + gap_len
            else:
                y = y0
                while y < y1:
                    draw.line([(x1, y), (x1, min(y + dash_len, y1))], fill=MONOSOMY_DASH_COLOR)
                    y += dash_len + gap_len
    else:
        draw.rectangle([1, 1, width - 2, height - 2], fill=PLACEHOLDER_COLOR)
    return img


def render_pair(
    copies: list[Image.Image],
    label: str,
    target_height: int,
    avg_w: int,
) -> Image.Image:
    """Compose up to 3 chromosome copies for one position with a label below.

    Copy 1 and 2 are the standard homolog pair. Copy 3 (trisomy) is placed
    adjacent with a red border. Missing copies get monosomy placeholders.
    """
    normalized = [normalize_crop(c, target_height) for c in copies]

    w1 = normalized[0].width if len(normalized) >= 1 else avg_w
    w2 = normalized[1].width if len(normalized) >= 2 else avg_w
    w3 = normalized[2].width if len(normalized) >= 3 else 0

    total_w = max(w1 + PAIR_GAP + w2 + (PAIR_GAP + w3 if w3 else 0), 20)
    total_h = target_height + LABEL_HEIGHT
    canvas = Image.new("RGB", (total_w, total_h), color=BG_COLOR)

    # First homolog (or monosomy placeholder)
    canvas.paste(normalized[0] if len(normalized) >= 1
                 else placeholder_image(w1, target_height, dashed=True), (0, 0))

    # Second homolog (or monosomy placeholder)
    x2 = w1 + PAIR_GAP
    canvas.paste(normalized[1] if len(normalized) >= 2
                 else placeholder_image(w2, target_height, dashed=True), (x2, 0))

    # Trisomy extra copy with red border highlight
    if len(normalized) >= 3:
        x3 = x2 + w2 + PAIR_GAP
        tri = normalized[2].copy()
        ImageDraw.Draw(tri).rectangle(
            [0, 0, tri.width - 1, tri.height - 1],
            outline=TRISOMY_BORDER_COLOR, width=TRISOMY_BORDER,
        )
        canvas.paste(tri, (x3, 0))

    # Chromosome label centered below the pair
    draw = ImageDraw.Draw(canvas)
    font = load_font(11)
    short = label.replace("chr", "").upper()
    try:
        bb = draw.textbbox((0, 0), short, font=font)
        text_w = bb[2] - bb[0]
    except AttributeError:
        text_w = int(draw.textlength(short, font=font))
    draw.text((max(0, (total_w - text_w) // 2), target_height + 6),
              short, fill=LABEL_COLOR, font=font)
    return canvas


def render_row(
    row_spec: list,
    grouped: dict[str, list[dict]],
    target_height: int,
) -> Image.Image:
    """Render one karyogram row as a horizontal PIL image strip.

    Handles separators, chromosome pairs, and group labels.
    """
    row_h = target_height + LABEL_HEIGHT
    font_group = load_font(13, bold=True)

    # First pass: measure total width
    pair_images: list[tuple[int, Image.Image]] = []
    group_annots: list[tuple[int, str]] = []
    sep_xs: list[int] = []
    current_x = 0

    for spec in row_spec:
        if spec == "|":
            sep_xs.append(current_x)
            current_x += SEPARATOR_GAP
            continue
        label = spec[0]
        copies_data = grouped.get(label, [])
        crops = [d["crop"] for d in copies_data if d.get("crop") is not None]
        avg_w = normalize_crop(crops[0], target_height).width if crops else target_height // 2
        pair_img = render_pair(crops, label, target_height, avg_w)
        pair_images.append((current_x, pair_img))
        if label in DENVER_GROUP_LABELS:
            group_annots.append((current_x, DENVER_GROUP_LABELS[label]))
        current_x += pair_img.width + POSITION_GAP

    strip = Image.new("RGB", (max(current_x, 1), row_h), color=BG_COLOR)
    draw = ImageDraw.Draw(strip)

    for gx, img in pair_images:
        strip.paste(img, (gx, 0))

    for gx, gname in group_annots:
        if gname != "sex":
            draw.text((gx, 1), gname, fill=GROUP_LABEL_COLOR, font=font_group)

    for sx in sep_xs:
        sep_x = sx - SEPARATOR_GAP // 2
        draw.line([(sep_x, 4), (sep_x, row_h - 4)], fill=SEPARATOR_COLOR, width=1)

    return strip


def render_grid(row_images: list[Image.Image]) -> Image.Image:
    """Stack row images vertically with spacing and outer margins."""
    if not row_images:
        return Image.new("RGB", (400, 200), color=BG_COLOR)
    max_w = max(img.width for img in row_images)
    total_h = sum(img.height for img in row_images) + ROW_VGAP * (len(row_images) - 1)
    canvas = Image.new("RGB", (max_w + 2 * MARGIN, total_h + 2 * MARGIN), color=BG_COLOR)
    y = MARGIN
    for img in row_images:
        canvas.paste(img, (MARGIN, y))
        y += img.height + ROW_VGAP
    return canvas
