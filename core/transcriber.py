import os
from core.config import get_transcriber_instance

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    transcriber = get_transcriber_instance()
    task = "translate" if language.lower() == "hinglish" else "transcribe"
    return transcriber.transcribe(chunk_path, task=task)
