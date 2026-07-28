#
# pyrefly: ignore [missing-import]
from PIL import Image 
import moondream as md
from .base import BaseVisionModel


class Moondream_3_1_Offline(BaseVisionModel):
    def __init__(self, model_name: str = "moondream3.1-9B-A2B"):
        super().__init__()
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading Moondream3.1 Offline model: {self.model_name} ...")
            self.model = md.vl(local=True, model=self.model_name)
            print("Moondream3.1 Offline model loaded.")

    def analyze(self, image_paths: list[dict], prompt: str = "") -> list[dict]:
        self._load_model()
        results = []
        for item in image_paths:
            timeStamp = item["timestamp"]
            image_path = item["image_path"]
            image = Image.open(image_path)
            result = self.model.query(
                image=image, 
                question=prompt
            )
            results.append({
                "timestamp": timeStamp,
                "caption": result["answer"]
            })
        return results


