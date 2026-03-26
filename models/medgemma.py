import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from models.base import BaseVLM


class MedGemmaBase(BaseVLM):
    """
    Shared base for MedGemma model variants.
    Subclasses set MODEL_ID to select the specific checkpoint.
    """

    MODEL_ID: str  # defined by subclass

    SYSTEM_PROMPT = "You are an expert radiologist."

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self.processor = None

    def load_model(self):
        print(f"Loading {self.MODEL_ID}...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print(f"{self.MODEL_ID} loaded.")

    def _build_messages(self, image: Image.Image, prompt: str) -> list:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 2048, **kwargs) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = self._build_messages(image, prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )
            generation = generation[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True)