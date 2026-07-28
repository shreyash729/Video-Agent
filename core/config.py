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
        hf_token = config.get("hf_token")
        model_kwargs = {"token": hf_token} if hf_token else {}
        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs=model_kwargs
        )
    elif mode == "online":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=config.get("hf_token")
        )
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")

def get_vision_model(config: dict):
    isSelected = config.get("is_Vision_Selected") or config.get("is_vision_selected") # if user wants image processing or not
    api_key = config.get("vision_api_key")
    if not isSelected:  # user does not want video/image vision processing
        return False
    mode = config.get("vision_mode")
    model = config.get("vision_model")
    if not mode:
        raise ValueError("Vision configuration is missing. Please set vision_mode.")
    
    if mode == "offline":
        if model == "moondream3.1-9B-A2B":
            from core.vision_models.Moondream_3_1_Offline import Moondream_3_1_Offline
            return Moondream_3_1_Offline()
        elif model == "Moondream-2":
            from core.vision_models.Moondream2Offline import Moondream2Offline
            return Moondream2Offline()
        elif model == "moondream/moondream3-preview":
            from core.vision_models.MoonDream3_Preview_offline import MoonDream3_Preview_offline
            return MoonDream3_Preview_offline()
        elif model == "Salesforce/blip-image-captioning-base":
            from core.vision_models.SalesForce_Blip_Base_Offline import BLIPImageCaptioningBaseOffline
            return BLIPImageCaptioningBaseOffline()
        elif model == "Salesforce/blip-image-captioning-large":
            from core.vision_models.SalesForce_Blip_Large_Offline import SalesForce_Blip_Large_Offline
            return SalesForce_Blip_Large_Offline()
        else:
            raise ValueError(f"Unknown vision model: {model}")
    elif mode == "online":
        if model in ["moondream 3", "moondream/moondream3-preview", "moondream"]:
            if not api_key:
                return False
            from core.vision_models.MoonDream3_Preview_online import MoonDream3_Preview_online
            return MoonDream3_Preview_online(api_key)
        else:
            return False
    else:
        return False

def get_llm(config: dict):
    if not config.get("provider") or not config.get("model") or not config.get("api_key"):
        raise ValueError("LLM configuration is missing. Please set provider, model and api_key.")

    return init_chat_model(
        model=config["model"],
        model_provider=config["provider"],
        api_key=config["api_key"],
        temperature=0.3
    )
