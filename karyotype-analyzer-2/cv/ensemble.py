"""5-strategy ensemble classifier for chromosome identification.

Combines Simple CNN, Siamese Contrastive, SRAS-Enhanced, VariFocal Fusion,
and Multi-Task strategies via weighted majority voting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from cv.classifier import ChromosomeClassifier, CHROMOSOME_TEMPLATES
from cv.ensemble_helpers import (
    strategy_simple_cnn,
    strategy_siamese,
    strategy_sras,
    strategy_varifocal,
    strategy_multitask,
)


class EnsembleClassifier:
    """5-strategy ensemble with weighted majority voting."""

    CHROMOSOME_TEMPLATES = CHROMOSOME_TEMPLATES

    def __init__(self):
        self.base_classifier = ChromosomeClassifier()
        self.strategy_weights = {
            "simple_cnn": 0.15,
            "siamese": 0.20,
            "sras": 0.20,
            "varifocal": 0.25,
            "multitask": 0.20,
        }

    def classify_ensemble(
        self,
        gray: np.ndarray,
        contours: List,
        boxes: List[Tuple],
        areas: List[float],
        band_profiles: List[np.ndarray],
        straightened: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], Dict]:
        """Run all 5 classification strategies and combine via weighted voting."""
        n = len(contours)
        if n == 0:
            return [], {"strategies": {}}

        total_area = sum(areas) if areas else 1
        bc = self.base_classifier

        s1 = strategy_simple_cnn(gray, contours, boxes, areas, total_area, bc)
        s2 = strategy_siamese(gray, contours, boxes, areas, band_profiles, total_area, bc)
        s3 = strategy_sras(gray, contours, boxes, areas, total_area, bc)
        s4 = strategy_varifocal(
            gray, contours, boxes, areas, band_profiles, straightened, total_area, bc
        )
        s5 = strategy_multitask(
            gray, contours, boxes, areas, band_profiles, straightened, total_area, bc
        )

        all_strategies = [
            ("simple_cnn", s1),
            ("siamese", s2),
            ("sras", s3),
            ("varifocal", s4),
            ("multitask", s5),
        ]

        final_results = []
        for i in range(n):
            vote_scores: Dict[str, float] = {}
            strategy_votes: Dict[str, str] = {}
            for strategy_name, strategy_results in all_strategies:
                if i >= len(strategy_results):
                    continue
                pred_class = strategy_results[i]["class"]
                confidence = strategy_results[i]["confidence"]
                weight = self.strategy_weights[strategy_name]
                vote_scores[pred_class] = (
                    vote_scores.get(pred_class, 0) + weight * confidence
                )
                strategy_votes[strategy_name] = pred_class

            if vote_scores:
                winner = max(vote_scores, key=vote_scores.get)
                winner_score = vote_scores[winner]
                total_weight = sum(vote_scores.values())
                normalized_conf = winner_score / total_weight if total_weight > 0 else 0
            else:
                winner = "?"
                normalized_conf = 0

            template = self.CHROMOSOME_TEMPLATES.get(
                winner, {"group": "?", "type": "unknown"}
            )
            top3 = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)[:3]

            final_results.append({
                "predicted_class": winner,
                "confidence": round(normalized_conf, 3),
                "denver_group": template["group"],
                "chromosome_type": template.get("type", "unknown"),
                "strategy_votes": strategy_votes,
                "vote_scores": {k: round(v, 3) for k, v in vote_scores.items()},
                "top3": [(c, round(s, 3)) for c, s in top3],
                "features": s1[i].get("features", {}) if i < len(s1) else {},
            })

        final_results = bc._refine_with_pairing(final_results)

        agreement_count = 0
        for i in range(n):
            votes = [sr[i]["class"] for _, sr in all_strategies if i < len(sr)]
            if len(set(votes)) == 1:
                agreement_count += 1

        metadata = {
            "strategies_used": list(self.strategy_weights.keys()),
            "weights": self.strategy_weights,
            "unanimous_agreement": agreement_count,
            "agreement_ratio": round(agreement_count / n, 3) if n > 0 else 0,
            "total_chromosomes": n,
        }
        return final_results, metadata
