from abc import ABC, abstractmethod
from PIL import Image

class BaseVLM(ABC):
    """
    Abstract base class for all VLMs.
    Every model must implement load_model() and generate().
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load model weights and processor/tokenizer."""
        pass

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Run inference on a single image and prompt.

        Args:
            image:  PIL Image (grayscale or RGB).
            prompt: Text prompt to condition generation.
            **kwargs: Model-specific generation parameters.

        Returns:
            Generated report text as a string.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"