from faster_whisper import WhisperModel
from .base import BaseTranscriber

class FasterWhisperTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "medium"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading faster-whisper model: {self.model_name} ...")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
            print("faster-whisper model loaded.")

    def transcribe(self, chunk_path: str, task: str = "transcribe") -> str | list[dict]:
        self._load_model()
        segments, info = self.model.transcribe(chunk_path, task=task, vad_filter=True)
        transcript_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        # full_text = ""
        # for segment in segments:
        #     full_text += segment.text + " "
        # return full_text.strip()

        return transcript_segments
