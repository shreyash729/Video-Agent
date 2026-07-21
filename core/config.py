import os
from langchain.chat_models import init_chat_model

def get_transcriber_instance(config: dict):
    mode = config.get("transcription_mode")
    model = config.get("transcription_model")
    
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
        return HFInferenceTranscriber(model_name=model, hf_token=config.get("hf_token"))
    else:
        raise ValueError(f"Unknown transcription mode: {mode}")

def get_embedding_instance(config: dict):
    mode = config.get("embedding_mode")
    model = config.get("embedding_model")
    
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
            model=model,
            huggingfacehub_api_token=config.get("hf_token")
        )
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")

def get_llm(config: dict):
    if not config.get("provider") or not config.get("model") or not config.get("api_key"):
        raise ValueError("LLM configuration is missing. Please set provider, model and api_key.")
        
    # Inject API key securely for this specific invocation without mutating global env vars if possible.
    # However, some langchain models require env vars. We will temporarily set it.
    key_map = {
        "openai": "OPENAI_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "mistralai": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    env_key = key_map.get(config.get("provider"))
    if env_key:
        os.environ[env_key] = config.get("api_key")
        
    hf_token = config.get("hf_token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        
    return init_chat_model(
        model=config["model"],
        model_provider=config["provider"],
        temperature=0.3
    )
