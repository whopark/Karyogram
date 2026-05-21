"""FLUX.1 image-to-image karyogram enhancement module.

Sends a raw PIL karyogram to fal.ai FLUX.1 dev img2img API to produce
a visually polished, clinically presentable version. The enhancement is
optional — if the API call fails, the raw karyogram is returned unchanged.
"""

import io
import logging
import os
from typing import Optional

from PIL import Image

log = logging.getLogger(__name__)

try:
    import fal_client
    _FAL_AVAILABLE = True
except ImportError:
    _FAL_AVAILABLE = False

DEFAULT_PROMPT = (
    "clinical karyogram, clean scientific diagram, high-resolution "
    "chromosome banding patterns, professional medical illustration, "
    "white background, sharp precise lines"
)
DEFAULT_STRENGTH = 0.65


class EnhancementError(Exception):
    """Raised when FLUX.1 enhancement fails."""


def get_fal_key() -> Optional[str]:
    """Retrieve fal.ai API key from FAL_KEY environment variable."""
    return os.environ.get("FAL_KEY")


def _upload_image(pil_image: Image.Image) -> str:
    """Convert PIL Image to data URI for the FLUX.1 API."""
    import base64  # noqa: PLC0415
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _download_image(url: str) -> Image.Image:
    """Download an image from a URL and return as PIL Image.

    Only HTTPS URLs are accepted to prevent SSRF attacks.
    """
    import urllib.request  # noqa: PLC0415
    if not url.startswith("https://"):
        raise EnhancementError(f"Unexpected URL scheme (expected https): {url[:80]}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


# @AX:NOTE: [AUTO] FLUX.1 img2img enhancement — external API dependency, requires FAL_KEY
def enhance_karyogram(
    raw_image: Image.Image,
    api_key: str,
    strength: float = DEFAULT_STRENGTH,
    prompt: Optional[str] = None,
) -> Image.Image:
    """Enhance a raw karyogram using FLUX.1 dev img2img.

    Args:
        raw_image: PIL Image of the raw karyogram from generate_karyogram().
        api_key: fal.ai API key (FAL_KEY).
        strength: Transformation intensity (0-1). Lower preserves more layout.
        prompt: Text prompt for the enhancement. Defaults to clinical karyogram prompt.

    Returns:
        Enhanced PIL Image.

    Raises:
        EnhancementError: If the API call fails for any reason.
    """
    if not _FAL_AVAILABLE:
        raise EnhancementError("fal-client package is not installed. Run: pip install fal-client")

    prev_key = os.environ.get("FAL_KEY")
    os.environ["FAL_KEY"] = api_key
    enhancement_prompt = prompt or DEFAULT_PROMPT

    try:
        image_url = _upload_image(raw_image)
        log.info("Submitted karyogram for enhancement")

        result = fal_client.subscribe(
            "fal-ai/flux/dev/image-to-image",
            arguments={
                "image_url": image_url,
                "prompt": enhancement_prompt,
                "strength": strength,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
            },
        )

        output_url = result["images"][0]["url"]
        enhanced = _download_image(output_url)
        log.info("Enhancement complete, output size: %s", enhanced.size)
        return enhanced

    except EnhancementError:
        raise
    except Exception as exc:
        log.warning("FLUX.1 enhancement failed: %s", exc)
        raise EnhancementError(f"FLUX.1 API call failed: {exc}") from exc
    finally:
        if prev_key is not None:
            os.environ["FAL_KEY"] = prev_key
        else:
            os.environ.pop("FAL_KEY", None)
