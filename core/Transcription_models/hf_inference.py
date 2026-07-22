import os
from huggingface_hub import InferenceClient
from .base import BaseTranscriber

class HFInferenceTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "openai/whisper-large-v3-turbo", hf_token: str = None):
        self.model_name = model_name
        self.hf_token = hf_token
        self.client = None

    def _load_client(self):
        if self.client is None:
            token = self.hf_token or os.getenv("HF_TOKEN")
            if not token:
                raise ValueError("HF_TOKEN environment variable is not set. Online transcription requires it.")
            
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=token,
            )

    def transcribe(self, chunk_path: str, task: str = "transcribe") -> str | list[dict]:
        self._load_client()
        result = self.client.automatic_speech_recognition(chunk_path, model=self.model_name, return_timestamps=True, )
        transcript_segments = [
            {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
            for chunk in result["chunks"]
        ]
        return transcript_segments

