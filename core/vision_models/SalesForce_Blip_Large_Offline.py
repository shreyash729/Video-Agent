from transformers import BlipProcessor, BlipForConditionalGeneration
from .base import BaseVisionModel
from PIL import Image 

class SalesForce_Blip_Large_Offline(BaseVisionModel):
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large"):
        super().__init__()
        self.processor = None
        self.model = None
        self.model_name = model_name
        
    def _load_model(self):
        if self.processor is None:
            print(f"Loading model: {self.model_name} ...")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            if self.gpu:
                self.model = self.model.to("cuda")
            print("BLIP model loaded.")
    
    def analyze(self, image_paths: list[dict], prompt: str = "") -> list[dict]:
        self._load_model()
        results = []
        for item in image_paths:
            timeStamp = item["timestamp"]
            image_path = item["image_path"]
            raw_image = Image.open(image_path).convert('RGB')
            if prompt:
                inputs = self.processor(raw_image, prompt, return_tensors="pt")
            else:
                inputs = self.processor(raw_image, return_tensors="pt")
            if self.gpu:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = self.model.generate(**inputs)
            results.append({
                "timestamp": timeStamp,
                "caption": self.processor.decode(out[0], skip_special_tokens=True)
            })
        return results


