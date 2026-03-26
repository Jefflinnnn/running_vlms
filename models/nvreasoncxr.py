import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.base import BaseVLM

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

PROMPTS = {
    "reasoning": "Write an example findings section for the CXR. Please reason step by step, and put your final answer within \\boxed{{}}.",
    "instruct":  "Write an example findings section for the CXR.",
}

# ──────────────────────────────────────────────
# NV-Reason-CXR-3B Model
# ──────────────────────────────────────────────

class NVReasonCXR(BaseVLM):
    """
    ** EXPERIMENTAL **

    NVIDIA NV-Reason-CXR-3B implementation based on Qwen2.5-VL.
    HuggingFace: nvidia/NV-Reason-CXR-3B

    Trained on MIMIC-CXR, ChestXRay14, and CheXpert.
    Supports two inference modes:
        - 'reasoning': Step by step reasoning with boxed final answer
        - 'instruct':  Direct report generation

    Note: Requires weights downloaded via:
        huggingface-cli download nvidia/NV-Reason-CXR-3B

    Note: Uses qwen_vl_utils.process_vision_info for correct image preprocessing.
        Install via: pip install qwen-vl-utils

    Note: The default basic usage guide on HF breaks in this pipeline for some reason.
        The workaround is using Qwen2_5_VLForConditionalGeneration instead of
        AutoModelForImageTextToText, and applying the chat template manually
        via the processor with tokenize=False.
    """

    MODEL_ID = "nvidia/NV-Reason-CXR-3B"

    def __init__(self, device: str = "cuda", mode: str = "instruct"):
        """
        Args:
            device: Device to run inference on.
            mode:   One of 'reasoning' or 'instruct'.
        """
        super().__init__(device=device)
        if mode not in PROMPTS:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(PROMPTS.keys())}")
        self.mode = mode
        self.processor = None

    def load_model(self):
        """Load NV-Reason-CXR weights and processor."""
        if self.model is not None:
            print("Model already loaded, skipping.")
            return

        print(f"Loading {self.MODEL_ID} in '{self.mode}' mode...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        print("NV-Reason-CXR loaded.")

    def _build_messages(self, image: Image.Image, prompt: str) -> list:
        """Format image and prompt into the expected chat message structure."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

    def generate(
        self,
        image: Image.Image,
        prompt: str | None = None,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        Generate a radiology report for a given chest X-ray image.

        Args:
            image:          PIL Image (any mode, converted to RGB internally).
            prompt:         Text prompt. Defaults to the mode's preset prompt.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs:       Additional arguments forwarded to model.generate().

        Returns:
            Generated report text as a string.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() before generate().")

        prompt = prompt or PROMPTS[self.mode]
        image = image.convert("RGB")

        messages = self._build_messages(image, prompt)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

if __name__ == "__main__":
    from data.hf_loader import hf_login
    from data.dataset import CheXpertDataset

    hf_login()

    dataset = CheXpertDataset(subset="findings", split="valid")
    image, findings_text = dataset[0]

    image.show()

    # Test instruct mode
    model = NVReasonCXR(mode="instruct")
    model.load_model()
    output_instruct = model.generate(image)
    print(f"Ground truth:\n{findings_text}")
    print(f"\nNV-Reason-CXR instruct output:\n{output_instruct}")

    # Test reasoning mode
    model_reasoning = NVReasonCXR(mode="reasoning")
    model_reasoning.load_model()
    output_reasoning = model_reasoning.generate(image)
    print(f"\nNV-Reason-CXR reasoning output:\n{output_reasoning}")