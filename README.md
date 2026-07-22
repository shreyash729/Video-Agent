# 🎥 Video Agent

Video Agent is a powerful, AI-driven video and meeting assistant designed to process YouTube videos or local audio/video files, generate intelligent summaries, extract key action items, and allow you to directly chat with the video transcript using a sophisticated Retrieval-Augmented Generation (RAG) pipeline.

---

## ✨ Features

- **Multi-Source Input:** Process any YouTube URL or upload local media files directly.
- **Intelligent Summarization:** Automatically generates comprehensive summaries, key decisions, action items, and open questions using state-of-the-art LLMs.
- **Interactive Chat (RAG):** Ask questions about the video! The app builds a local vector database of the transcript and uses RAG to accurately answer your queries based *only* on the video content.
- **Hybrid AI Pipeline:**
  - **Offline Mode:** Run transcription (`faster-whisper`) and embedding generation entirely on your local machine using your own CPU/GPU for maximum privacy and zero API costs.
  - **Online Mode:** Offload the heavy lifting to Hugging Face Inference endpoints.
- **Bot Bypass:** Built-in cookie-extraction logic to securely bypass YouTube's strict anti-bot protections.
- **Provider Agnostic:** Easily swap between OpenAI, Mistral, Google GenAI, and Anthropic for the core LLM processing.

---

## 🧠 Models & Capabilities

### 1. Large Language Models (Summarization & Chat)
Powered by **LangChain**, the app allows you to configure your preferred LLM provider via the settings UI.
- OpenAI (GPT-4o, GPT-3.5)
- Google GenAI (Gemini 1.5 Pro/Flash)
- Mistral AI
- Anthropic (Claude 3.5 Sonnet)

### 2. Transcription Models (Speech-to-Text)
- **Local (faster-whisper):** `large-v3`, `medium`, `small`, `base`, `openai/whisper-large-v3-turbo`
- **Online (Hugging Face):** `openai/whisper-large-v3-turbo`, `openai/whisper-large-v3`

### 3. Embedding Models (RAG Vectorization)
- **Local (Sentence Transformers):** `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `google/embeddinggemma-300m`, `LiquidAI/LFM2.5-Embedding-350M`
- **Online (Hugging Face):** `google/embeddinggemma-300m`, `LiquidAI/LFM2.5-Embedding-350M`, `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`

---

## 🚀 Local Installation & Usage

### Prerequisites
1. **Python 3.10, 3.11, or 3.12** *(Note: Python 3.13+ is not supported due to the removal of the `audioop` module required by `pydub`).*
2. **FFmpeg** installed and added to your system PATH.
3. (Optional but recommended) Google Chrome or Microsoft Edge installed (used for extracting YouTube authentication cookies).

### Quick Start

**1. Clone the repository and navigate to the directory:**
```bash
git clone https://github.com/your-username/video-agent.git
cd video-agent
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv .venv
# Activate on Windows:
.\.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate

# Install core dependencies (Required for all usages):
pip install -r requirements.txt

# (OPTIONAL) If you want to use heavy offline models locally, you MUST also install:
pip install -r requirements_local.txt
```

**3. Configure your Environment (Optional):**
Create a `.env` file in the root directory. 
```env
ALLOW_LOCAL_MODEL="true"
```
*(If you deploy this app publicly, you can set `ALLOW_LOCAL_MODEL="false"` to prevent users from crashing your server by forcing it to run heavy offline models).*

**4. Start the Application:**
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`).

---

## ⚙️ Configuration Guide

When you first launch the app, click the **Settings (Gear Icon)** in the top right to configure your models:

1. **LLM Settings:** Select your preferred provider (e.g., OpenAI) and enter your API key. This key is used for the summarization and chatbot reasoning.
2. **Transcription Settings:** Choose **Offline** to download and run the Whisper model on your own hardware, or **Online** to ping the Hugging Face API (requires a free HF Token).
3. **Embedding Settings:** Choose your vectorization model. Offline mode will use ChromaDB locally.

*Note: All API keys are stored securely in your local environment and are never exposed in the frontend.*

---

## 🛠️ Troubleshooting

**Issue:** `yt_dlp.utils.DownloadError: ERROR: Sign in to confirm you’re not a bot.`
**Solution:** YouTube is throttling your IP. The app is programmed to automatically grab your browser cookies to bypass this. Just ensure you are logged into YouTube on Chrome or Edge, and make sure your browser is closed when you run the pipeline so the cookie database isn't locked!

**Issue:** `ModuleNotFoundError: No module named 'audioop'`
**Solution:** You are running Python 3.13 or newer. You must downgrade to Python 3.12 or older.
