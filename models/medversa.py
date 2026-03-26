import sys
import tempfile
import torch
from pathlib import Path
from PIL import Image
from models.base import BaseVLM

# ──────────────────────────────────────────────
# MedVersa Path Setup
# ──────────────────────────────────────────────

# Point to the HuggingFace cached MedVersa code
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MEDVERSA_SNAPSHOT = next(HF_CACHE.glob("models--hyzhou--MedVersa/snapshots/*/"))
sys.path.insert(0, str(MEDVERSA_SNAPSHOT))

# ──────────────────────────────────────────────
# Default Values
# ──────────────────────────────────────────────

DEFAULT_CONTEXT = "Age: unknown.\nGender: unknown.\nIndication: unknown.\nComparison: None."

DEFAULT_PARAMS = {
    "num_beams": 1,
    "do_sample": True,
    "min_length": 1,
    "top_p": 0.9,
    "repetition_penalty": 1,
    "length_penalty": 1,
    "temperature": 0.1,
}

# ──────────────────────────────────────────────
# MedVersa Model
# ──────────────────────────────────────────────

class MedVersa(BaseVLM):
    """
    MedVersa (MedOmni) implementation.
    HuggingFace: hyzhou/MedVersa

    Note: Requires MedVersa weights downloaded via:
        hf download hyzhou/MedVersa
    """

    MODEL_ID = "hyzhou/MedVersa"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)

    def load_model(self):
        """Load MedVersa weights using its custom registry."""
        print(f"Loading {self.MODEL_ID}...")
        from utils import registry
        model_cls = registry.get_model_class("medomni")
        self.model = model_cls.from_pretrained(self.MODEL_ID).to(self.device).eval()
        print("MedVersa loaded.")

    def _build_context(self, context: dict | None) -> str:
        """
        Build the patient context string.

        Args:
            context: Optional dict with keys 'age', 'gender', 'indication', 'comparison'.
                     Any missing keys fall back to 'unknown'.
                     Pass None to use the full default context.

        Example:
            {
                "age": "30-40",
                "gender": "F",
                "indication": "Dyspnea, PICC line placement.",
                "comparison": "None."
            }

        Returns:
            Formatted context string.
        """
        if context is None:
            return DEFAULT_CONTEXT

        return (
            f"Age: {context.get('age', 'unknown')}.\n"
            f"Gender: {context.get('gender', 'unknown')}.\n"
            f"Indication: {context.get('indication', 'unknown')}.\n"
            f"Comparison: {context.get('comparison', 'None')}."
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str = "How would you characterize the findings from <img0>?",
        context: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a radiology report for a given image and prompt.

        Args:
            image:    PIL Image in RGB format.
            prompt:   Text prompt. Use <img0> to reference the image.
            context:  Optional dict with patient metadata keys:
                      'age', 'gender', 'indication', 'comparison'.
                      Falls back to DEFAULT_CONTEXT if None.
            **kwargs: Override DEFAULT_PARAMS generation parameters.

        Returns:
            Generated report text as a string.
        """
        from utils import generate_predictions

        # MedVersa expects file paths, save PIL image to a temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        context_str = self._build_context(context)
        params = {**DEFAULT_PARAMS, **kwargs}

        _, _, output_text = generate_predictions(
            self.model,
            [image_path],
            context_str,
            prompt,
            "cxr",
            "report generation",
            self.device,    # positional, must come before **params
            **params,
        )

        return output_text


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data.hf_loader import hf_login
    from data.dataset import CheXpertDataset

    hf_login()

    dataset = CheXpertDataset(subset="findings", split="valid")
    image, findings_text = dataset[0]

    model = MedVersa()
    model.load_model()

    # No context
    output = model.generate(image)
    print(f"Ground truth:\n{findings_text}")
    print(f"\nMedVersa output (no context):\n{output}")

    # With context
    output_with_context = model.generate(
        image,
        context={
            "age": "50-60",
            "gender": "M",
            "indication": "Shortness of breath.",
            "comparison": "None."
        }
    )
    print(f"\nMedVersa output (with context):\n{output_with_context}")