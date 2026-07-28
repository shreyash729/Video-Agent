from abc import ABC, abstractmethod
import torch

class BaseVisionModel(ABC):
    def __init__(self):
        self.gpu = True if torch.cuda.is_available() else False
    @abstractmethod
    def analyze(self, image_paths: list[dict], prompt: str = "") -> list[dict]:
        """Analyzes the given image and returns the text."""
        pass
