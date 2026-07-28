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
        try:
            result = self.client.automatic_speech_recognition(chunk_path, model=self.model_name, return_timestamps=True)
        except TypeError:
            # Fallback for newer/different huggingface_hub SDK method signatures
            result = self.client.automatic_speech_recognition(chunk_path, model=self.model_name)

        if isinstance(result, str):
            return [{"start": 0.0, "end": 0.0, "text": result.strip()}]
        
        if isinstance(result, dict):
            chunks = result.get("chunks")
            if chunks and isinstance(chunks, list):
                segments = []
                for chunk in chunks:
                    ts = chunk.get("timestamp") or [0.0, 0.0]
                    start = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 and ts[0] is not None else 0.0
                    end = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 and ts[1] is not None else 0.0
                    segments.append({"start": start, "end": end, "text": chunk.get("text", "").strip()})
                return segments
            return [{"start": 0.0, "end": 0.0, "text": result.get("text", "").strip()}]

        return [{"start": 0.0, "end": 0.0, "text": str(result).strip()}]

