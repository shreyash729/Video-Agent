import os
import json
from langchain.chat_models import init_chat_model

CONFIG_FILE = "config.json"

current_config = {
    "provider": None,
    "model": None,
    "api_key": None,
    "transcription_mode": None,
    "transcription_model": None,
    "hf_token": None,
    "embedding_mode": None,
    "embedding_model": None
}

def load_config_from_disk():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
                current_config.update(saved)
                _apply_env_vars()
        except Exception as e:
            print(f"Error loading config: {e}")

def save_config_to_disk():
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(current_config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

def _apply_env_vars():
    key_map = {
        "openai": "OPENAI_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "mistralai": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    provider = current_config.get("provider")
    api_key = current_config.get("api_key")
    if provider and api_key:
        env_key = key_map.get(provider)
        if env_key:
            os.environ[env_key] = api_key
            
    hf_token = current_config.get("hf_token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

def set_config(provider: str, model: str, api_key: str):
    current_config["provider"] = provider
    current_config["model"] = model
    if api_key:
        current_config["api_key"] = api_key
    _apply_env_vars()
    save_config_to_disk()

def set_transcription_config(mode: str, model: str, hf_token: str = None):
    current_config["transcription_mode"] = mode
    current_config["transcription_model"] = model
    if hf_token:
        current_config["hf_token"] = hf_token
    _apply_env_vars()
    save_config_to_disk()
    
def set_embedding_config(mode: str, model: str):
    current_config["embedding_mode"] = mode
    current_config["embedding_model"] = model
    save_config_to_disk()

def get_transcriber_instance():
    mode = current_config.get("transcription_mode")
    model = current_config.get("transcription_model")
    
    if not mode or not model:
        raise ValueError("Transcription configuration is missing. Please set mode and model.")
        
    if mode == "offline":
        if model in ["large-v3", "base", "small", "medium"]:
            from core.Transcription_models.faster_whisper_model import FasterWhisperTranscriber
            return FasterWhisperTranscriber(model_name=model)
        else:
            from core.Transcription_models.transformers_whisper import TransformersWhisperTranscriber
            return TransformersWhisperTranscriber(model_id=model)
    elif mode == "online":
        from core.Transcription_models.hf_inference import HFInferenceTranscriber
        return HFInferenceTranscriber(model_name=model)
    else:
        raise ValueError(f"Unknown transcription mode: {mode}")

def get_embedding_instance():
    mode = current_config.get("embedding_mode")
    model = current_config.get("embedding_model")
    
    if not mode or not model:
        raise ValueError("Embedding configuration is missing.")
        
    if mode == "offline":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True},
        )
    elif mode == "online":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model=model
        )
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")

def get_llm():
    if not current_config.get("provider") or not current_config.get("model"):
        raise ValueError("LLM configuration is missing. Please set provider and model in Configuration.")
    return init_chat_model(
        model=current_config["model"],
        model_provider=current_config["provider"],
        temperature=0.3
    )

# Load config on startup
load_config_from_disk()
