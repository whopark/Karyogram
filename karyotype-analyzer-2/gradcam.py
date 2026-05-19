"""CAM visualization for chromosome classification explainability.

Generates Class Activation Maps via weight projection (Zhou et al. 2016).
No backward pass required -- uses the linear relationship between GAP features
and classifier head weights to produce spatial heatmaps.
"""

import argparse
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _resolve_layer(model, dotted_name: str):
    """Resolve a nested module by dot-separated name (e.g. 'backbone.layer4')."""
    return reduce(getattr, dotted_name.split("."), model)


def generate_cam(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: str = "backbone.layer4",
) -> np.ndarray:
    """Produce a CAM heatmap for *target_class* via weight projection.

    Args:
        model: ChromosomeResNet18 instance (eval mode expected).
        input_tensor: (1, 3, H, W) tensor, already normalized.
        target_class: Class index in [0, 24).
        target_layer: Dot-path to the convolutional layer whose feature maps
            are used for the spatial heatmap.

    Returns:
        2-D numpy array (H, W) with values in [0, 1].
    """
    layer = _resolve_layer(model, target_layer)
    captured = {}

    def _hook(module, inp, out):  # noqa: ARG001
        captured["feats"] = out.detach()

    handle = layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_tensor)
        feats = captured["feats"]  # (1, C, fH, fW)

        # Compose the two linear layers into a single weight vector per class.
        # head[3].weight[c] @ head[0].weight -> shape (512,)
        w_proj = model.head[3].weight[target_class] @ model.head[0].weight
        w_proj = w_proj.detach()

        # Weighted combination of feature map channels.
        n_channels = feats.shape[1]
        cam = torch.zeros(feats.shape[2:], device=feats.device)
        for c in range(n_channels):
            cam += w_proj[c] * feats[0, c]

        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Upsample to input spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, fH, fW)
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
    finally:
        handle.remove()

    return cam


def overlay_heatmap(
    cam_map: np.ndarray,
    original_image: Image.Image,
    colormap: str = "inferno",
    alpha: float = 0.5,
    threshold_percentile: int = 80,
) -> Image.Image:
    """Blend a CAM heatmap onto *original_image*.

    Args:
        cam_map: (H, W) array in [0, 1] from :func:`generate_cam`.
        original_image: Source chromosome crop (PIL RGB Image).
        colormap: Matplotlib colormap name.
        alpha: Heatmap opacity (0 = invisible, 1 = opaque).
        threshold_percentile: Percentile below which CAM values are zeroed.

    Returns:
        PIL RGB Image with the heatmap overlay.
    """
    import matplotlib.cm as cm  # noqa: PLC0415

    # Apply threshold
    thresh = np.percentile(cam_map, threshold_percentile)
    cam_t = np.where(cam_map >= thresh, cam_map, 0.0)

    # Re-normalize
    cam_max = cam_t.max()
    if cam_max > 0:
        cam_t = cam_t / cam_max

    # Apply colormap -> RGBA float array
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(cam_t.astype(np.float32))  # (H, W, 4)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb).resize(
        original_image.size, Image.BILINEAR,
    )

    # Alpha-blend
    orig_arr = np.array(original_image.convert("RGB"), dtype=np.float32)
    heat_arr = np.array(heatmap_pil, dtype=np.float32)
    blended = alpha * heat_arr + (1 - alpha) * orig_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, "RGB")


def render_cam_grid(classifications, model, device, st_module):
    """Render a Streamlit grid of CAM overlays for classified chromosomes.

    Called from karyogram_ui.py as a single-line integration point.
    When *model* or *device* is None, the classifier is loaded via the
    cached ``_load_models`` helper (same cache as the main pipeline).

    Args:
        classifications: List of dicts with 'crop', 'label', 'confidence' keys.
        model: ChromosomeResNet18 instance in eval mode, or None to auto-load.
        device: torch device, or None to auto-detect.
        st_module: The ``streamlit`` module (passed to avoid import at module level).
    """
    from training.chromosome_model import build_eval_transform, IDX_TO_LABEL  # noqa: PLC0415

    if model is None or device is None:
        from ml_pipeline import load_classifier  # noqa: PLC0415
        info = load_classifier()
        if "error" in info:
            st_module.warning(f"Cannot generate CAM: {info['error']}")
            return
        model, device = info["model"], info["device"]

    transform = build_eval_transform()
    cols = st_module.columns(min(len(classifications), 6))

    for idx, cls_info in enumerate(classifications):
        crop = cls_info["crop"]
        label = cls_info.get("label", "?")
        class_idx = IDX_TO_LABEL.index(label) if label in IDX_TO_LABEL else 0
        tensor = transform(crop).unsqueeze(0).to(device)
        cam = generate_cam(model, tensor, class_idx)
        overlay = overlay_heatmap(cam, crop.convert("RGB").resize((128, 128)))
        col = cols[idx % len(cols)]
        col.image(overlay, caption=label, width=100)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CAM heatmap for a chromosome image.")
    parser.add_argument("--model", required=True, help="Path to classifier .pth weights")
    parser.add_argument("--image", required=True, help="Path to chromosome crop image")
    parser.add_argument("--target-class", type=int, default=0, help="Class index (0-23)")
    parser.add_argument("--target-layer", default="backbone.layer4", help="Dot-path to conv layer")
    parser.add_argument("--colormap", default="inferno", help="Matplotlib colormap name")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap opacity")
    parser.add_argument("--output", default="cam_overlay.png", help="Output image path")
    args = parser.parse_args()

    from training.chromosome_model import ChromosomeResNet18, build_eval_transform  # noqa: PLC0415

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = ChromosomeResNet18(pretrained=False)
    state = torch.load(args.model, map_location=dev, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    mdl.load_state_dict(state)
    mdl.to(dev).eval()

    img = Image.open(args.image).convert("RGB")
    tfm = build_eval_transform()
    inp = tfm(img).unsqueeze(0).to(dev)

    cam_result = generate_cam(mdl, inp, args.target_class, args.target_layer)
    result_img = overlay_heatmap(cam_result, img, args.colormap, args.alpha)
    result_img.save(args.output)
    print(f"CAM overlay saved to {args.output}")
