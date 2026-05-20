"""CV package: computer vision pipeline for chromosome detection.

Public API:
    ChromosomeDetector  -- orchestrates full detection pipeline
    ChromosomeClassifier -- 24-class morphometric classifier
    EnsembleClassifier  -- 5-strategy ensemble
"""

from cv.detector import ChromosomeDetector
from cv.classifier import ChromosomeClassifier
from cv.ensemble import EnsembleClassifier

__all__ = ["ChromosomeDetector", "ChromosomeClassifier", "EnsembleClassifier"]
