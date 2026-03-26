import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from models.base import BaseVLM

class MedGemma(BaseVLM):
    """
    MedGemma-1.5-4b-it implementation.
    HuggingFace: google/medgemma-1.5-4b-it
    """

    MODEL_ID = "google/medgemma-1.5-4b-it"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.processor = None

    def load_model(self):
        """Load MedGemma weights and processor."""
        print(f"Loading {self.MODEL_ID}...")
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("MedGemma loaded.")

    def _build_messages(self, image: Image.Image, prompt: str) -> list:
        """Format image and prompt into MedGemma's expected message structure."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """
        Generate a radiology report for a given image and prompt.

        Args:
            image:          PIL Image (grayscale or RGB).
            prompt:         Text prompt to condition generation.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs:       Additional generation parameters (e.g. temperature, top_p).

        Returns:
            Generated report text as a string.
        """
        messages = self._build_messages(image, prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )
            generation = generation[0][input_len:]  # strip input tokens from output

        return self.processor.decode(generation, skip_special_tokens=True)


if __name__ == "__main__":
    from data.hf_loader import hf_login
    from data.dataset import CheXpertDataset

    hf_login()

    # Load a sample image
    dataset = CheXpertDataset(subset="findings", split="valid")
    image, findings_text = dataset[0]

    # Run MedGemma
    model = MedGemma()
    model.load_model()

    prompt = "Generate a radiology report for this chest X-ray."
    output = model.generate(image, prompt)

    print(f"Ground truth:\n{findings_text}")
    print(f"\nMedGemma output:\n{output}")