"""VLM package: Vision-Language Model analysis providers.

Public API:
    KaryotypeAnalyzer      -- multi-provider dispatcher
    PrecisionClinicalLens  -- 6-stage sequential pipeline
"""

from vlm.analyzer import KaryotypeAnalyzer
from vlm.precision_lens import PrecisionClinicalLens

__all__ = ["KaryotypeAnalyzer", "PrecisionClinicalLens"]
