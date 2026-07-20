import os
from huggingface_hub import InferenceClient
from .base import BaseTranscriber

class HFInferenceTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "openai/whisper-large-v3-turbo"):
        self.model_name = model_name
        self.client = None

    def _load_client(self):
        if self.client is None:
            token = os.getenv("HF_TOKEN")
            if not token:
                raise ValueError("HF_TOKEN environment variable is not set. Online transcription requires it.")
            
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=token,
            )

    def transcribe(self, chunk_path: str, task: str = "transcribe") -> str:
        self._load_client()
        result = self.client.automatic_speech_recognition(chunk_path, model=self.model_name)
        # result is an object with text attribute
        return result.text.strip()
