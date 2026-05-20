"""Shared provider definitions, import guards, and availability flags.

Single source of truth for APIProvider enum and optional SDK imports.
All other modules import provider definitions from this module.
"""

from enum import Enum

# Computer Vision
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Vision-Language Model API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class APIProvider(Enum):
    OPENAI = "OpenAI GPT-4 Vision"
    ANTHROPIC = "Anthropic Claude Vision"
    GEMINI = "Google Gemini Vision"
    CONSENSUS = "Multi-Model Consensus"
    CV_VLM = "CV + VLM (Hybrid)"
    TWO_STAGE = "Two-Stage Pipeline (CV + VLM)"
    PRECISION_LENS = "Precision Clinical Lens (6-Stage)"
    YOLO_KARYOGRAM = "YOLO Karyogram (ML Pipeline)"
    MOCK = "Demo Mode (No API)"
