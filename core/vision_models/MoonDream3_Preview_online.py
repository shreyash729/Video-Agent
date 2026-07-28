# pyrefly: ignore [missing-import]
from PIL import Image 
import moondream as md
from .base import BaseVisionModel

class MoonDream3_Preview_online(BaseVisionModel):
    def __init__(self, api: str = ""):
        super().__init__()
        self.model = md.vl(api_key=api)

    def analyze(self, image_paths: list[dict], prompt: str = "") -> list[dict]:
        results = []
        q = prompt if prompt else "Describe this image scene in detail."
        for item in image_paths:
            timeStamp = item["timestamp"]
            image_path = item["image_path"]
            image = Image.open(image_path)
            if hasattr(self.model, "query"):
                res = self.model.query(image=image, question=q, reasoning=False)
                caption = res.get("answer", str(res)) if isinstance(res, dict) else str(res)
            elif hasattr(self.model, "caption"):
                res = self.model.caption(image=image)
                caption = res.get("caption", str(res)) if isinstance(res, dict) else str(res)
            else:
                caption = ""
            results.append({
                "timestamp": timeStamp,
                "caption": caption
            })
        return results
