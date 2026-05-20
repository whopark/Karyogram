"""Classifier helper functions: centromere estimation, super-resolution
banding enhancement, and type compatibility scoring."""

import numpy as np
from typing import Optional, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


def estimate_centromere_index(
    gray: np.ndarray, contour, box: Tuple
) -> float:
    """Estimate centromere index = p-arm length / total length.
    The centromere is at the narrowest constriction point."""
    x, y, w, h = box
    if h < 5 or w < 3:
        return 0.5

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - np.array([x, y])
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    widths = []
    for row in range(h):
        row_pixels = np.sum(mask[row, :] > 0)
        widths.append(row_pixels)

    if not widths or max(widths) == 0:
        return 0.5

    widths_arr = np.array(widths, dtype=np.float64)
    if len(widths_arr) > 5:
        kernel_size = min(5, len(widths_arr) // 2) | 1
        widths_smooth = np.convolve(
            widths_arr, np.ones(kernel_size) / kernel_size, mode="same"
        )
    else:
        widths_smooth = widths_arr

    margin = max(1, int(h * 0.15))
    search_region = widths_smooth[margin : h - margin]
    if len(search_region) == 0:
        return 0.5

    min_idx = np.argmin(search_region) + margin
    centromere_index = min_idx / h
    return float(np.clip(centromere_index, 0.1, 0.9))


def super_resolve_banding(
    gray: np.ndarray,
    contour,
    box: Tuple,
    existing_profile: Optional[np.ndarray],
    super_res_target: int = 64,
) -> np.ndarray:
    """Super-resolution enhancement for low-quality banding patterns."""
    x, y, w, h = box
    num_bins = 32

    if existing_profile is not None and h >= super_res_target:
        return existing_profile

    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return existing_profile if existing_profile is not None else np.zeros(num_bins)

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - np.array([x, y])
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    if h < super_res_target:
        scale = super_res_target / h
        new_w = max(3, int(w * scale))
        roi_upscaled = cv2.resize(
            roi, (new_w, super_res_target), interpolation=cv2.INTER_CUBIC
        )
        mask_upscaled = cv2.resize(
            mask, (new_w, super_res_target), interpolation=cv2.INTER_NEAREST
        )
        blurred = cv2.GaussianBlur(roi_upscaled, (0, 0), 1.0)
        roi_sharp = cv2.addWeighted(roi_upscaled, 1.5, blurred, -0.5, 0)
    else:
        roi_sharp = roi
        mask_upscaled = mask

    sr_h, sr_w = roi_sharp.shape[:2]
    profile = np.zeros(sr_h, dtype=np.float64)
    for row in range(sr_h):
        row_mask = (
            mask_upscaled[row, :] if row < mask_upscaled.shape[0] else np.zeros(sr_w)
        )
        masked_pixels = roi_sharp[row, :][row_mask > 0]
        if len(masked_pixels) > 0:
            profile[row] = np.mean(masked_pixels)

    if len(profile) > 1:
        resampled = np.interp(
            np.linspace(0, len(profile) - 1, num_bins),
            np.arange(len(profile)),
            profile,
        )
    else:
        resampled = np.zeros(num_bins)

    norm = np.linalg.norm(resampled)
    return resampled / norm if norm > 0 else resampled


def type_compatibility(measured_ci: float, expected_type: str) -> float:
    """Score how well the measured centromere index matches the expected type."""
    if expected_type == "metacentric":
        return max(0, 1.0 - abs(measured_ci - 0.45) * 4)
    elif expected_type == "submetacentric":
        return max(0, 1.0 - abs(measured_ci - 0.33) * 4)
    elif expected_type == "acrocentric":
        return max(0, 1.0 - abs(measured_ci - 0.20) * 4)
    return 0.5
