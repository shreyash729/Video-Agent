from .base import BaseVisionModel
from PIL import Image 
import torch
from transformers import AutoModelForCausalLM

class MoonDream3_Preview_offline(BaseVisionModel):
    def __init__(self, model_name: str = "moondream/moondream3-preview"):
        super().__init__()
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            print(f"Loading Moondream3_Preview_offline model: {self.model_name} ...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            if self.gpu:
                self.model = self.model.to("cuda")
            self.model.compile()
            print("Moondream3_Preview_offline model loaded.")

    def analyze(self, image_paths: list[dict], prompt: str = "") -> list[dict]:
        self._load_model()
        results = []
        q = prompt if prompt else "Describe this image scene in detail."
        for item in image_paths:
            timeStamp = item["timestamp"]
            image_path = item["image_path"]
            image = Image.open(image_path)
            res = self.model.query(image=image, question=q, reasoning=False)
            caption = res["answer"]
            results.append({
                "timestamp": timeStamp,
                "caption": caption
            })
        return results  
