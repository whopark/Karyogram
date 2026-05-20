"""Cluster router: classifies chromosome overlap types and dispatches
to the appropriate segmentation strategy."""

import numpy as np
from typing import Dict, List, Tuple

from providers import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2

from cv.cluster_router_helpers import (
    route_a_touching,
    route_b_one_overlap,
    route_c_multi_overlap,
)


class ClusterRouter:
    """Pre-routing mechanism that classifies contour clusters by overlap
    type and routes each to the optimal segmentation pipeline."""

    class OverlapType:
        ISOLATED = "isolated"
        TOUCHING = "touching"
        ONE_OVERLAP = "one_overlap"
        MULTI_OVERLAP = "multi_overlap"

    def __init__(self):
        self.adjacency_dilate_px = 5
        self.overlap_dilate_px = 3
        self.area_overlap_threshold = 0.08
        self.multi_overlap_min = 3

    def classify_clusters(
        self,
        gray: np.ndarray,
        binary_mask: np.ndarray,
        contours: List,
        boxes: List[Tuple],
        areas: List[float],
    ) -> List[Dict]:
        """Classify each contour by its overlap relationship with neighbors."""
        if not CV2_AVAILABLE or not contours:
            return []

        h, w = gray.shape
        n = len(contours)

        masks = []
        for c in contours:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m, [c], -1, 255, -1)
            masks.append(m)

        adjacency = [[False] * n for _ in range(n)]
        overlap_areas = [[0.0] * n for _ in range(n)]
        kernel_adj = np.ones(
            (self.adjacency_dilate_px * 2 + 1, self.adjacency_dilate_px * 2 + 1),
            np.uint8,
        )

        dilated_masks = [cv2.dilate(m, kernel_adj, iterations=1) for m in masks]

        for i in range(n):
            for j in range(i + 1, n):
                intersection = cv2.bitwise_and(dilated_masks[i], masks[j])
                inter_pixels = np.sum(intersection > 0)
                if inter_pixels > 0:
                    adjacency[i][j] = True
                    adjacency[j][i] = True
                direct_overlap = cv2.bitwise_and(masks[i], masks[j])
                overlap_px = np.sum(direct_overlap > 0)
                min_a = min(areas[i], areas[j]) if areas[i] > 0 and areas[j] > 0 else 1
                overlap_ratio = overlap_px / min_a
                overlap_areas[i][j] = overlap_ratio
                overlap_areas[j][i] = overlap_ratio

        # BFS connected components
        visited = [False] * n
        cluster_ids = [-1] * n
        cluster_id = 0

        def bfs(start):
            queue = [start]
            members = []
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                cluster_ids[node] = cluster_id
                members.append(node)
                for neighbor in range(n):
                    if adjacency[node][neighbor] and not visited[neighbor]:
                        queue.append(neighbor)
            return members

        for i in range(n):
            if not visited[i]:
                bfs(i)
                cluster_id += 1

        results = []
        for i in range(n):
            neighbors = [j for j in range(n) if adjacency[i][j]]
            overlapping_neighbors = [
                j
                for j in neighbors
                if overlap_areas[i][j] > self.area_overlap_threshold
            ]
            touching_only = [j for j in neighbors if j not in overlapping_neighbors]

            if not neighbors:
                overlap_type = self.OverlapType.ISOLATED
            elif not overlapping_neighbors:
                overlap_type = self.OverlapType.TOUCHING
            elif len(overlapping_neighbors) == 1:
                overlap_type = self.OverlapType.ONE_OVERLAP
            else:
                overlap_type = self.OverlapType.MULTI_OVERLAP

            max_overlap = max(
                (overlap_areas[i][j] for j in neighbors), default=0.0
            )
            results.append(
                {
                    "index": i,
                    "overlap_type": overlap_type,
                    "neighbor_indices": neighbors,
                    "overlapping_neighbors": overlapping_neighbors,
                    "touching_neighbors": touching_only,
                    "max_overlap_ratio": max_overlap,
                    "cluster_id": cluster_ids[i],
                }
            )
        return results

    def route_and_segment(
        self,
        gray: np.ndarray,
        binary_mask: np.ndarray,
        contours: List,
        boxes: List[Tuple],
        areas: List[float],
        min_area: int,
        segmenter,
    ) -> Tuple[List, List[Tuple], List[float], Dict]:
        """Route each cluster to the appropriate segmentation pipeline."""
        if not CV2_AVAILABLE or not contours:
            return contours, boxes, areas, {"routing": "empty"}

        classifications = self.classify_clusters(
            gray, binary_mask, contours, boxes, areas
        )
        if not classifications:
            return contours, boxes, areas, {"routing": "no_classifications"}

        processed = [False] * len(contours)
        final_contours: List = []
        final_boxes: List[Tuple] = []
        final_areas: List[float] = []
        route_stats = {
            "isolated": 0, "touching": 0, "one_overlap": 0, "multi_overlap": 0,
            "route_a_splits": 0, "route_b_splits": 0, "route_c_splits": 0,
        }

        cluster_groups: Dict[int, List] = {}
        for info in classifications:
            cid = info["cluster_id"]
            cluster_groups.setdefault(cid, []).append(info)

        OT = self.OverlapType
        for cid, group in cluster_groups.items():
            indices = [info["index"] for info in group]
            types = [info["overlap_type"] for info in group]
            has_multi = OT.MULTI_OVERLAP in types
            has_one = OT.ONE_OVERLAP in types
            has_touching = OT.TOUCHING in types

            if len(indices) == 1 and types[0] == OT.ISOLATED:
                idx = indices[0]
                final_contours.append(contours[idx])
                final_boxes.append(boxes[idx])
                final_areas.append(areas[idx])
                processed[idx] = True
                route_stats["isolated"] += 1
            elif has_multi:
                cr = route_c_multi_overlap(
                    gray, binary_mask, contours, boxes, areas, indices, min_area, segmenter
                )
                for c, b, a in cr:
                    final_contours.append(c)
                    final_boxes.append(b)
                    final_areas.append(a)
                for idx in indices:
                    processed[idx] = True
                route_stats["multi_overlap"] += len(indices)
                route_stats["route_c_splits"] += max(0, len(cr) - len(indices))
            elif has_one:
                cr = route_b_one_overlap(
                    gray, binary_mask, contours, boxes, areas, indices, min_area, segmenter
                )
                for c, b, a in cr:
                    final_contours.append(c)
                    final_boxes.append(b)
                    final_areas.append(a)
                for idx in indices:
                    processed[idx] = True
                route_stats["one_overlap"] += len(indices)
                route_stats["route_b_splits"] += max(0, len(cr) - len(indices))
            elif has_touching:
                cr = route_a_touching(
                    gray, binary_mask, contours, boxes, areas, indices, min_area, segmenter
                )
                for c, b, a in cr:
                    final_contours.append(c)
                    final_boxes.append(b)
                    final_areas.append(a)
                for idx in indices:
                    processed[idx] = True
                route_stats["touching"] += len(indices)
                route_stats["route_a_splits"] += max(0, len(cr) - len(indices))
            else:
                for idx in indices:
                    final_contours.append(contours[idx])
                    final_boxes.append(boxes[idx])
                    final_areas.append(areas[idx])
                    processed[idx] = True
                    route_stats["isolated"] += 1

        for i in range(len(contours)):
            if not processed[i]:
                final_contours.append(contours[i])
                final_boxes.append(boxes[i])
                final_areas.append(areas[i])

        metadata = {
            "routing": "cluster_router",
            "total_clusters": len(cluster_groups),
            "route_stats": route_stats,
        }
        return final_contours, final_boxes, final_areas, metadata
