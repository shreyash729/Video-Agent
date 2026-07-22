import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .base import BaseTranscriber

class TransformersWhisperTranscriber(BaseTranscriber):
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        self.model_id = model_id
        self.pipe = None

    def _load_model(self):
        if self.pipe is None:
            print(f"Loading transformers model: {self.model_id} ...")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(self.model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
                chunk_length_s=30,
                ignore_warning=True,
            )
            print("transformers model loaded.")

    def transcribe(self, chunk_path: str, task: str = "transcribe") -> str | list[dict]:
        self._load_model()
        generate_kwargs = {"task": task}
        result = self.pipe(chunk_path, generate_kwargs=generate_kwargs, return_timestamps=True)
        transcript_segments = [
            {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
            for chunk in result["chunks"]
        ]
        return transcript_segments
