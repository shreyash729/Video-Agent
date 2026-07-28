from PIL import Image 
import torch
from transformers import AutoModelForCausalLM
from .base import BaseVisionModel

# --- MONKEY PATCH FOR TRANSFORMERS VERSION COMPATIBILITY ---
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        val = getattr(self, "_tied_weights_keys", {})
        return val if isinstance(val, dict) else {}
    return _orig_getattr(self, name)

torch.nn.Module.__getattr__ = _patched_getattr
# ------------------------------------------------------------

class Moondream2Offline(BaseVisionModel):
    def __init__(self, model_name: str = "vikhyatk/moondream2", revision: str = "2025-06-21"):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            print(f"Loading Moondream2 model: {self.model_name} ...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True
            )
            if self.gpu:
                self.model = self.model.to("cuda")
            print("Moondream2 model loaded.")
    
    def analyze(self, image_paths: list[dict], prompt: str = "") -> list[dict]:
        self._load_model() 
        results = []
        q = prompt if prompt else "Describe this image scene in detail."
        for item in image_paths:
            timeStamp = item["timestamp"]
            image_path = item["image_path"]
            image = Image.open(image_path)
            res = self.model.query(image=image, question=q)
            caption = res["answer"] if isinstance(res, dict) and "answer" in res else str(res)
            results.append({
                "timestamp": timeStamp,
                "caption": caption
            })
        return results