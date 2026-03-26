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
# CheXOne Model
# ──────────────────────────────────────────────

class CheXOne(BaseVLM):
    """
    CheXOne implementation based on Qwen2.5-VL.
    HuggingFace: StanfordAIMI/CheXOne

    Supports two inference modes:
        - 'reasoning': Step by step reasoning with boxed final answer
        - 'instruct':  Direct report generation

    Note: Requires weights downloaded via:
        hf download StanfordAIMI/CheXOne
    """

    MODEL_ID = "StanfordAIMI/CheXOne"

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
        """Load CheXOne weights and processor."""
        print(f"Loading {self.MODEL_ID} in '{self.mode}' mode...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        print("CheXOne loaded.")

    def _build_messages(self, image: Image.Image, prompt: str) -> list:
        """Format image and prompt into CheXOne's expected message structure."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate(
        self,
        image: Image.Image,
        prompt: str | None = None,
        max_new_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """
        Generate a radiology report for a given image.

        Args:
            image:          PIL Image in RGB format.
            prompt:         Optional custom prompt. Defaults to the mode's preset prompt.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs:       Additional generation parameters.

        Returns:
            Generated report text as a string.
        """
        prompt = prompt or PROMPTS[self.mode]
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

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]  # batch size is always 1 here

if __name__ == "__main__":
    from data.hf_loader import hf_login
    from data.dataset import CheXpertDataset

    hf_login()

    dataset = CheXpertDataset(subset="findings", split="valid")
    image, findings_text = dataset[0]

    # Test instruct mode
    model = CheXOne(mode="instruct")
    model.load_model()
    output_instruct = model.generate(image)
    print(f"Ground truth:\n{findings_text}")
    print(f"\nCheXOne instruct output:\n{output_instruct}")

    # Test reasoning mode
    model_reasoning = CheXOne(mode="reasoning")
    model_reasoning.load_model()
    output_reasoning = model_reasoning.generate(image)
    print(f"\nCheXOne reasoning output:\n{output_reasoning}")