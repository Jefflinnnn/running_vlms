import sys
import tempfile
import torch
from pathlib import Path

# ──────────────────────────────────────────────
# MedVersa Path Setup
# ──────────────────────────────────────────────

# Point to the HuggingFace cached MedVersa code
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MEDVERSA_SNAPSHOT = sorted(HF_CACHE.glob("models--hyzhou--MedVersa/snapshots/*/"))[-1]
sys.path.insert(0, str(MEDVERSA_SNAPSHOT))

# ──────────────────────────────────────────────
# Transformers Compatibility Patches
# ──────────────────────────────────────────────
# MedVersa's bundled Qformer.py was written against transformers ~4.35, which
# exported several utilities (apply_chunking_to_forward, prune_linear_layer,
# prune_conv1d_layer, find_pruneable_heads_and_indices) directly from
# transformers.modeling_utils. In later versions of transformers, these were
# either relocated to transformers.pytorch_utils or removed from the public API
# entirely. Since Qformer.py imports them at module level with hard-coded paths,
# it raises an ImportError before any model code runs. Rather than modifying the
# MedVersa snapshot directly, which would break reproducibility and HF cache
# integrity, i patch the missing names back onto transformers.modeling_utils
# at process startup, before the MedVersa import chain fires. Functions still
# present in pytorch_utils are aliased across; functions removed entirely are
# re-implemented from scratch using their original logic.
# Sigh...

import torch
import transformers.modeling_utils as _mu
import transformers.pytorch_utils as _pu

# ── Patch 1: names moved from modeling_utils to pytorch_utils ──
_MOVED_TO_PYTORCH_UTILS = [
    "apply_chunking_to_forward",
    "prune_linear_layer",
]
for _name in _MOVED_TO_PYTORCH_UTILS:
    if not hasattr(_mu, _name) and hasattr(_pu, _name):
        setattr(_mu, _name, getattr(_pu, _name))

# ── Patch 2: find_pruneable_heads_and_indices removed entirely ──
if not hasattr(_mu, "find_pruneable_heads_and_indices"):
    def _find_pruneable_heads_and_indices(
        heads: list,
        n_heads: int,
        head_size: int,
        already_pruned_heads: set,
    ) -> tuple:
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    _mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

# ── Patch 3: prune_conv1d_layer removed entirely ──
if not hasattr(_mu, "prune_conv1d_layer"):
    def _prune_conv1d_layer(layer, index: torch.LongTensor, dim: int = 1):
        """Prune a Conv1D layer to keep only entries in index on dim."""
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if dim == 0:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        # Conv1D in transformers stores weight as (in, out) unlike nn.Linear
        from transformers.pytorch_utils import Conv1D
        new_layer = Conv1D(new_size[1], new_size[0])
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
        return new_layer

    _mu.prune_conv1d_layer = _prune_conv1d_layer

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
from PIL import Image
from models.base import BaseVLM
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
        try:
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
        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            Path(image_path).unlink(missing_ok=True)

        return output_text

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