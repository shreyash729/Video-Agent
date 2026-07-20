from abc import ABC, abstractmethod

class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, chunk_path: str, task: str = "transcribe") -> str:
        """Transcribes the given audio chunk and returns the text."""
        pass
