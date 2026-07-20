from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import threading
import time
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

# Keep backend core imports unchanged
from utils.audio_processer import process_input
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import ask_question, format_docs
from core.vector_store import get_retriever
from core import vector_store
from core.transcriber import transcribe_chunk
from core.config import set_config, current_config, set_transcription_config, set_embedding_config, get_llm

# pyright: ignore [reportMissingImports]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma

app = Flask(__name__)
CORS(app)

# Shared pipeline state
state_lock = threading.Lock()
pipeline_state = {
    "running": False,
    "source": None,
    "language": "english",
    "transcript": None,
    "title": None,
    "summary": None,
    "action_items": None,
    "decisions": None,
    "questions": None,
    "rag_chain": None,
    "pipeline_steps": {
        "audio_extract": {"status": "pending", "details": "Extracting Audio"},
        "audio_chunk": {"status": "pending", "details": "Creating Audio Chunks"},
        "transcribe_chunks": {"status": "pending", "details": "Transcribing chunks [0/0]", "progress": 0, "total": 0},
        "transcribe_combine": {"status": "pending", "details": "Combining chunks"},
        "summarize_llm": {"status": "pending", "details": "Generating Summary"},
        "rag_chunking": {"status": "pending", "details": "Chunking transcript"},
        "rag_embedding": {"status": "pending", "details": "Creating Embeddings [0/0]", "progress": 0, "total": 0},
        "rag_db": {"status": "pending", "details": "Initializing Chroma DB"},
        "rag_complete": {"status": "pending", "details": "Completed"}
    },
}

def set_step(key, status, details=None, progress=None, total=None, activity=None):
    with state_lock:
        pipeline_state['pipeline_steps'][key]['status'] = status
        if details is not None:
            pipeline_state['pipeline_steps'][key]['details'] = details
        if progress is not None:
            pipeline_state['pipeline_steps'][key]['progress'] = progress
        if total is not None:
            pipeline_state['pipeline_steps'][key]['total'] = total

import sys
import re

class StreamCatcher:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self._text_buffer = ""
    def write(self, s):
        self.original_stream.write(s)
        self._text_buffer += s
        while '\r' in self._text_buffer or '\n' in self._text_buffer:
            r_idx = self._text_buffer.find('\r')
            n_idx = self._text_buffer.find('\n')
            if r_idx != -1 and n_idx != -1:
                idx = min(r_idx, n_idx)
            else:
                idx = max(r_idx, n_idx)
            
            line = self._text_buffer[:idx].strip()
            self._text_buffer = self._text_buffer[idx+1:]
            
            if '%' in line and ('Downloading' in line or 'Fetching' in line or '|' in line or 'B/s' in line or 'eta' in line.lower()):
                match = re.search(r'(\d+)%', line)
                if match:
                    pct = int(match.group(1))
                    with state_lock:
                        is_audio = pipeline_state.get('pipeline_steps', {}).get('audio_extract', {}).get('status') == 'active'
                        is_transcribing = pipeline_state.get('pipeline_steps', {}).get('transcribe_chunks', {}).get('status') == 'active'
                        is_embedding = pipeline_state.get('pipeline_steps', {}).get('rag_embedding', {}).get('status') == 'active'
                        is_rag_db = pipeline_state.get('pipeline_steps', {}).get('rag_db', {}).get('status') == 'active'
                        
                        target_key = None
                        if is_audio:
                            target_key = 'audio_extract'
                            prefix = "Downloading Video"
                        elif is_transcribing:
                            target_key = 'transcribe_chunks'
                            prefix = "Downloading Model"
                        elif is_embedding:
                            target_key = 'rag_embedding'
                            prefix = "Downloading Model"
                        elif is_rag_db:
                            target_key = 'rag_db'
                            prefix = "Downloading Model"
                            
                        if target_key:
                            if pct == 100:
                                pipeline_state['pipeline_steps'][target_key]['details'] = f"{prefix} Completed. Processing..."
                            else:
                                pipeline_state['pipeline_steps'][target_key]['details'] = f"{prefix}... {pct}%"
    def flush(self):
        self.original_stream.flush()

def pipeline_worker(source, language):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StreamCatcher(old_stdout)
    sys.stderr = StreamCatcher(old_stderr)
    try:
        with state_lock:
            pipeline_state['running'] = True
            pipeline_state['source'] = source
            pipeline_state['language'] = language

        # Audio processing
        set_step('audio_extract', 'active', details="Extracting Audio...")
        chunks = process_input(source)
        set_step('audio_extract', 'done')
        
        set_step('audio_chunk', 'active', details="Creating Audio Chunks...")
        set_step('audio_chunk', 'done')

        # Transcription
        total_chunks = len(chunks)
        set_step('transcribe_chunks', 'active', details=f"Transcribing chunks [0/{total_chunks}]", progress=0, total=total_chunks)
        
        full_transcript = ""
        for i, chunk in enumerate(chunks):
            set_step('transcribe_chunks', 'active', details=f"Transcribing chunks [{i+1}/{total_chunks}]", progress=i+1, total=total_chunks)
            text = transcribe_chunk(chunk, language=language)
            full_transcript += text + " "
        
        full_transcript = full_transcript.strip()
        set_step('transcribe_chunks', 'done')
        
        set_step('transcribe_combine', 'active', activity="Combining Transcript...")
        with state_lock:
            pipeline_state['transcript'] = full_transcript
            pipeline_state['title'] = generate_title(full_transcript)
        set_step('transcribe_combine', 'done')

        # Run summarization and RAG creation in parallel
        def do_summary():
            set_step('summarize_llm', 'active', activity="Generating Summary...")
            try:
                summary = summarize(full_transcript)
                with state_lock:
                    pipeline_state['summary'] = summary
            except Exception:
                traceback.print_exc()
            set_step('summarize_llm', 'done')

        def do_rag():
            try:
                set_step('rag_chunking', 'active', details="Chunking Transcript for RAG...")
                import time
                time.sleep(1)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                txt_chunks = splitter.split_text(full_transcript)
                docs = [Document(page_content=c, metadata={'chunk_index': i}) for i, c in enumerate(txt_chunks)]
                set_step('rag_chunking', 'done')

                set_step('rag_db', 'active', details="Initializing Chroma DB...")
                time.sleep(1)
                embeddings = vector_store.get_embeddings()
                unique_col = f"{vector_store.COLLECTION_NAME}_{int(time.time())}"
                vs = Chroma(collection_name=unique_col, embedding_function=embeddings, persist_directory=vector_store.CHROMA_DIR)
                set_step('rag_db', 'done')

                tot_rag = len(docs)
                set_step('rag_embedding', 'active', details=f"Creating Embeddings [0/{tot_rag}]", progress=0, total=tot_rag)
                
                for i, doc in enumerate(docs):
                    vs.add_documents([doc])
                    set_step('rag_embedding', 'active', details=f"Creating Embeddings [{i+1}/{tot_rag}]", progress=i+1, total=tot_rag)
                
                set_step('rag_embedding', 'done')
                
                # Build the RAG Chain
                retriever = vector_store.get_retriever(vs, k=4)
                llm = get_llm()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert video AI assistant. Answer the user's question based ONLY on the video transcript context provided below.\n\nIf the answer is not found in the context, say: \"I could not find this information in the meeting transcript.\"\n\nAlways be concise and precise. If quoting someone, mention it clearly.\n\nContext from video transcript:\n{context}"),
                    ("human", "{question}"),
                ])
                rag_chain = ({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                
                with state_lock:
                    pipeline_state['rag_chain'] = rag_chain
                
                set_step('rag_complete', 'done', activity="RAG Ready")

            except Exception:
                traceback.print_exc()

        t1 = threading.Thread(target=do_summary, daemon=True)
        t2 = threading.Thread(target=do_rag, daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Extract action items / decisions / questions (can run after summary)
        with state_lock:
            pipeline_state['action_items'] = extract_action_items(full_transcript)
            pipeline_state['decisions'] = extract_key_decisions(full_transcript)
            pipeline_state['questions'] = extract_questions(full_transcript)
            pipeline_state['current_activity'] = "Completed"

    except Exception as e:
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        with state_lock:
            pipeline_state['running'] = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():
    if request.content_type and 'multipart/form-data' in request.content_type:
        language = request.form.get('language', 'english')
        source_type = request.form.get('source_type')
        if source_type == 'url':
            source = request.form.get('source')
        else:
            file = request.files.get('file')
            if not file or file.filename == '':
                return jsonify({'ok': False, 'error': 'No file selected'})
            os.makedirs('downloads', exist_ok=True)
            filepath = os.path.join('downloads', file.filename)
            file.save(filepath)
            source = filepath
    else:
        data = request.get_json() or {}
        source = data.get('source')
        language = data.get('language', 'english')

    with state_lock:
        if pipeline_state['running']:
            return jsonify({'ok': False, 'error': 'Pipeline already running'})
        
        # Reset state
        pipeline_state.update({
            'running': True,
            'source': source,
            'language': language,
            'transcript': None,
            'title': None,
            'summary': None,
            'action_items': None,
            'decisions': None,
            'questions': None,
            'rag_chain': None,
            'current_activity': "Starting..."
        })
        for k in pipeline_state['pipeline_steps'].keys():
            pipeline_state['pipeline_steps'][k]['status'] = 'pending'
            if 'progress' in pipeline_state['pipeline_steps'][k]:
                pipeline_state['pipeline_steps'][k]['progress'] = 0

    thread = threading.Thread(target=pipeline_worker, args=(source, language), daemon=True)
    thread.start()
    return jsonify({'ok': True})


@app.route('/status')
def status():
    with state_lock:
        out = {
            'running': pipeline_state['running'],
            'source': pipeline_state['source'],
            'language': pipeline_state['language'],
            'transcript': bool(pipeline_state['transcript']),
            'summary': pipeline_state['summary'],
            'title': pipeline_state['title'],
            'pipeline_steps': pipeline_state['pipeline_steps'],
            'current_activity': pipeline_state['current_activity'],
            'transcribe_model': "small",
            'rag_embedding_model': "default",
        }
    return jsonify(out)

@app.route('/config', methods=['GET'])
def get_config_route():
    allow_local = os.environ.get('ALLOW_LOCAL_MODEL', 'true').lower() == 'true'
    repo = os.environ.get('GITHUB_REPO', 'https://github.com/your-username/video-agent')
    
    response_data = current_config.copy()
    response_data['allow_local_model'] = allow_local
    response_data['github_repo'] = repo
    return jsonify(response_data)

@app.route('/config', methods=['POST'])
def set_config_route():
    data = request.get_json() or {}
    provider = data.get('provider')
    model = data.get('model')
    api_key = data.get('api_key')
    
    # The config API can optionally omit some keys if we're only updating embeddings etc.
    # But since the UI will send all state in the final step or independently, we can process what's provided.
    
    transcription_mode = data.get('transcription_mode')
    embedding_mode = data.get('embedding_mode')
    
    # Enforce local model restrictions
    allow_local = os.environ.get('ALLOW_LOCAL_MODEL', 'true').lower() == 'true'
    if not allow_local:
        if transcription_mode == 'offline' or embedding_mode == 'offline':
            return jsonify({'ok': False, 'error': 'Local execution is disabled on this server. Please use online huggingface inference mode, or run the app locally on your own machine.'})
            
    if api_key == "********" or not api_key:
        api_key = current_config.get('api_key')
        
    if provider and model and api_key:
        from langchain.chat_models import init_chat_model
        import os
        key_map = {
            "openai": "OPENAI_API_KEY",
            "google_genai": "GOOGLE_API_KEY",
            "mistralai": "MISTRAL_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        env_key = key_map.get(provider)
        if env_key:
            os.environ[env_key] = api_key
            
        try:
            test_llm = init_chat_model(model=model, model_provider=provider, temperature=0.3)
            test_llm.invoke("Hello")
        except Exception as e:
            return jsonify({'ok': False, 'error': f"Model validation failed: {str(e)}"})
            
        set_config(provider, model, api_key)
    
    transcription_mode = data.get('transcription_mode')
    transcription_model = data.get('transcription_model')
    hf_token = data.get('hf_token')
    
    if hf_token == "********" or not hf_token:
        hf_token = current_config.get('hf_token')
    
    if transcription_mode and transcription_model:
        set_transcription_config(transcription_mode, transcription_model, hf_token)
        
    embedding_mode = data.get('embedding_mode')
    embedding_model = data.get('embedding_model')
    
    if embedding_mode and embedding_model:
        set_embedding_config(embedding_mode, embedding_model)
        
    return jsonify({'ok': True})


@app.route('/summary')
def get_summary():
    with state_lock:
        return jsonify({'summary': pipeline_state['summary']})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json() or {}
    q = data.get('question')
    if not q:
        return jsonify({'ok': False, 'error': 'no question provided'})
    with state_lock:
        rag = pipeline_state.get('rag_chain')
    if rag is None:
        return jsonify({'ok': False, 'error': 'RAG not ready'})
    try:
        answer = ask_question(rag, q)
        return jsonify({'ok': True, 'answer': answer})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
