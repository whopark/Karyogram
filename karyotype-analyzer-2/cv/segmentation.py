"""Segmentation matrix for overlapping chromosome separation.

Two-path segmentation: semantic pixel classification with stitching-based
overlap recovery, and instance segmentation via marker-controlled watershed.
"""

import numpy as np
from typing import Dict, List, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2

from cv.segmentation_helpers import stitch_from_overlap


class SegmentationMatrix:
    """Two-path segmentation: semantic + instance for chromosome separation."""

    def __init__(self):
        self.overlap_intensity_factor = 1.3
        self.morph_kernel_size = 3
        self.dist_transform_threshold = 0.35
        self.gradient_weight = 0.7

    # -- Path 1: Semantic Segmentation ----------------------------------
    def semantic_segmentation(
        self, gray: np.ndarray, binary_mask: np.ndarray
    ) -> Dict:
        """Classify each pixel as background(0), single-chromosome(1),
        or overlap(2)."""
        if not CV2_AVAILABLE:
            return {"error": "OpenCV not available"}

        h, w = gray.shape
        fg_mask = (binary_mask > 0).astype(np.uint8)
        fg_pixels = gray[fg_mask == 1]

        if len(fg_pixels) == 0:
            return {
                "semantic_map": np.zeros((h, w), dtype=np.uint8),
                "overlap_mask": np.zeros((h, w), dtype=np.uint8),
                "overlap_pixel_count": 0,
                "single_pixel_count": 0,
            }

        fg_mean = float(np.mean(fg_pixels))
        fg_std = float(np.std(fg_pixels))

        semantic_map = np.zeros((h, w), dtype=np.uint8)
        semantic_map[fg_mask == 1] = 1

        overlap_threshold = fg_mean - fg_std * 0.8
        overlap_candidates = (gray < overlap_threshold) & (fg_mask == 1)

        overlap_raw = overlap_candidates.astype(np.uint8) * 255
        kernel = np.ones(
            (self.morph_kernel_size, self.morph_kernel_size), np.uint8
        )
        overlap_cleaned = cv2.morphologyEx(
            overlap_raw, cv2.MORPH_OPEN, kernel, iterations=2
        )
        overlap_cleaned = cv2.morphologyEx(
            overlap_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        min_overlap_area = max(50, int(h * w * 0.0002))
        contours_ov, _ = cv2.findContours(
            overlap_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        overlap_final = np.zeros((h, w), dtype=np.uint8)
        for c in contours_ov:
            if cv2.contourArea(c) >= min_overlap_area:
                cv2.drawContours(overlap_final, [c], -1, 255, -1)

        semantic_map[overlap_final > 0] = 2

        return {
            "semantic_map": semantic_map,
            "overlap_mask": overlap_final,
            "overlap_pixel_count": int(np.sum(overlap_final > 0)),
            "single_pixel_count": int(np.sum(semantic_map == 1)),
        }

    # -- Path 2: Instance Segmentation ----------------------------------
    def instance_segmentation(
        self, gray: np.ndarray, binary_mask: np.ndarray, min_area: int
    ) -> Dict:
        """Marker-controlled watershed for per-instance chromosome masks."""
        if not CV2_AVAILABLE:
            return {"instances": [], "count": 0}

        h, w = gray.shape
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        if dist_transform.max() == 0:
            return {"instances": [], "count": 0}

        _, sure_fg = cv2.threshold(
            dist_transform,
            self.dist_transform_threshold * dist_transform.max(),
            255,
            0,
        )
        sure_fg = np.uint8(sure_fg)
        kernel_small = np.ones((2, 2), np.uint8)
        sure_fg = cv2.morphologyEx(
            sure_fg, cv2.MORPH_OPEN, kernel_small, iterations=1
        )

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        gradient_mag = (
            gradient_mag / (gradient_mag.max() + 1e-8) * 255
        ).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        if len(gray.shape) == 2:
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color_img = gray.copy()

        grad_3ch = cv2.cvtColor(gradient_mag, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(
            color_img, 1.0 - self.gradient_weight, grad_3ch, self.gradient_weight, 0
        )

        markers_ws = np.int32(markers)
        cv2.watershed(blended, markers_ws)

        instances = []
        for label_id in range(2, num_markers + 1):
            instance_mask = np.uint8(markers_ws == label_id) * 255
            inst_contours, _ = cv2.findContours(
                instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for ic in inst_contours:
                area = cv2.contourArea(ic)
                if area >= min_area * 0.4:
                    ix, iy, iw, ih = cv2.boundingRect(ic)
                    instances.append(
                        {
                            "contour": ic,
                            "bbox": (ix, iy, iw, ih),
                            "area": area,
                            "mask": instance_mask,
                            "label_id": label_id,
                        }
                    )

        return {
            "instances": instances,
            "count": len(instances),
            "markers_used": num_markers - 1,
            "watershed_labels": markers_ws,
        }

    # -- Combined Pipeline ----------------------------------------------
    def segment_and_separate(
        self,
        gray: np.ndarray,
        binary_mask: np.ndarray,
        contours: List,
        boxes: List[Tuple],
        areas: List[float],
        min_area: int,
    ) -> Tuple[List, List[Tuple], List[float], Dict]:
        """Run both segmentation paths and merge results."""
        metadata: Dict = {
            "semantic_overlap_pixels": 0,
            "instance_count": 0,
            "stitched_chromosomes": 0,
            "segmentation_method": "none",
        }

        sem_result = self.semantic_segmentation(gray, binary_mask)
        overlap_mask = sem_result.get("overlap_mask", np.zeros_like(gray))
        overlap_pixels = sem_result.get("overlap_pixel_count", 0)
        metadata["semantic_overlap_pixels"] = overlap_pixels

        inst_result = self.instance_segmentation(gray, binary_mask, min_area)
        metadata["instance_count"] = inst_result.get("count", 0)

        has_significant_overlap = overlap_pixels > (
            gray.shape[0] * gray.shape[1] * 0.001
        )

        if has_significant_overlap and contours:
            stitched = stitch_from_overlap(
                gray, sem_result["semantic_map"], overlap_mask, contours, min_area
            )
            metadata["stitched_chromosomes"] = len(stitched)
            if stitched:
                overlap_dilated = cv2.dilate(
                    overlap_mask, np.ones((5, 5), np.uint8), iterations=2
                )
                new_contours: List = []
                new_boxes: List[Tuple] = []
                new_areas: List[float] = []
                for c, b, a in zip(contours, boxes, areas):
                    c_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                    cv2.drawContours(c_mask, [c], -1, 255, -1)
                    intersection = cv2.bitwise_and(c_mask, overlap_dilated)
                    if np.sum(intersection) < a * 0.3:
                        new_contours.append(c)
                        new_boxes.append(b)
                        new_areas.append(a)
                for sc, sb, sa in stitched:
                    new_contours.append(sc)
                    new_boxes.append(sb)
                    new_areas.append(sa)
                metadata["segmentation_method"] = "semantic_stitching"
                return new_contours, new_boxes, new_areas, metadata

        inst_count = inst_result.get("count", 0)
        if 38 <= inst_count <= 52 and abs(inst_count - 46) < abs(
            len(contours) - 46
        ):
            inst_contours = [i["contour"] for i in inst_result["instances"]]
            inst_boxes = [i["bbox"] for i in inst_result["instances"]]
            inst_areas = [i["area"] for i in inst_result["instances"]]
            metadata["segmentation_method"] = "instance_watershed"
            return inst_contours, inst_boxes, inst_areas, metadata

        metadata["segmentation_method"] = "passthrough"
        return contours, boxes, areas, metadata
