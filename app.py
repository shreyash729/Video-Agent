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
from core.config import get_llm

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

# Multi-tenant job state
state_lock = threading.Lock()
jobs = {}
thread_local = threading.local()

def get_initial_job_state(source, language, config):
    return {
        "running": False,
        "source": source,
        "language": language,
        "config": config,
        "transcript": None,
        "title": None,
        "summary": None,
        "action_items": None,
        "decisions": None,
        "questions": None,
        "rag_chain": None,
        "current_activity": "Starting...",
        "text_buffer": "",
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
        }
    }

def set_step(job_id, key, status, details=None, progress=None, total=None, activity=None):
    with state_lock:
        if job_id not in jobs: return
        job_state = jobs[job_id]
        job_state['pipeline_steps'][key]['status'] = status
        if details is not None:
            job_state['pipeline_steps'][key]['details'] = details
        if progress is not None:
            job_state['pipeline_steps'][key]['progress'] = progress
        if total is not None:
            job_state['pipeline_steps'][key]['total'] = total
        if activity is not None:
            job_state['current_activity'] = activity

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
                    job_id = getattr(thread_local, 'job_id', None)
                    if job_id:
                        with state_lock:
                            if job_id in jobs:
                                job_state = jobs[job_id]
                                is_audio = job_state.get('pipeline_steps', {}).get('audio_extract', {}).get('status') == 'active'
                                is_transcribing = job_state.get('pipeline_steps', {}).get('transcribe_chunks', {}).get('status') == 'active'
                                is_embedding = job_state.get('pipeline_steps', {}).get('rag_embedding', {}).get('status') == 'active'
                                is_rag_db = job_state.get('pipeline_steps', {}).get('rag_db', {}).get('status') == 'active'
                                
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
                                        job_state['pipeline_steps'][target_key]['details'] = f"{prefix} Completed. Processing..."
                                    else:
                                        job_state['pipeline_steps'][target_key]['details'] = f"{prefix}... {pct}%"
    def flush(self):
        self.original_stream.flush()

def pipeline_worker(job_id, source, language, config):
    thread_local.job_id = job_id
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StreamCatcher(old_stdout)
    sys.stderr = StreamCatcher(old_stderr)
    try:
        with state_lock:
            if job_id not in jobs: return
            jobs[job_id]['running'] = True

        # Audio processing
        set_step(job_id, 'audio_extract', 'active', details="Extracting Audio...")
        chunks = process_input(source, job_id)
        set_step(job_id, 'audio_extract', 'done')
        
        set_step(job_id, 'audio_chunk', 'active', details="Creating Audio Chunks...")
        set_step(job_id, 'audio_chunk', 'done')

        # Transcription
        total_chunks = len(chunks)
        set_step(job_id, 'transcribe_chunks', 'active', details=f"Transcribing chunks [0/{total_chunks}]", progress=0, total=total_chunks)
        
        full_transcript = ""
        for i, chunk in enumerate(chunks):
            set_step(job_id, 'transcribe_chunks', 'active', details=f"Transcribing chunks [{i+1}/{total_chunks}]", progress=i+1, total=total_chunks)
            text = transcribe_chunk(chunk, config, language=language)
            full_transcript += text + " "
        
        full_transcript = full_transcript.strip()
        set_step(job_id, 'transcribe_chunks', 'done')
        
        set_step(job_id, 'transcribe_combine', 'active', activity="Combining Transcript...")
        with state_lock:
            jobs[job_id]['transcript'] = full_transcript
            jobs[job_id]['title'] = generate_title(full_transcript, config)
        set_step(job_id, 'transcribe_combine', 'done')

        # Run summarization and RAG creation in parallel
        def do_summary():
            set_step(job_id, 'summarize_llm', 'active', activity="Generating Summary...")
            try:
                summary = summarize(full_transcript, config)
                with state_lock:
                    jobs[job_id]['summary'] = summary
            except Exception:
                traceback.print_exc()
            set_step(job_id, 'summarize_llm', 'done')

        def do_rag():
            try:
                set_step(job_id, 'rag_chunking', 'active', details="Chunking Transcript for RAG...")
                import time
                time.sleep(1)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                txt_chunks = splitter.split_text(full_transcript)
                docs = [Document(page_content=c, metadata={'chunk_index': i}) for i, c in enumerate(txt_chunks)]
                set_step(job_id, 'rag_chunking', 'done')

                set_step(job_id, 'rag_db', 'active', details="Initializing Chroma DB...")
                time.sleep(1)
                embeddings = vector_store.get_embeddings(config)
                unique_col = f"{vector_store.COLLECTION_NAME}_{int(time.time())}_{job_id[:8]}"
                vs = Chroma(collection_name=unique_col, embedding_function=embeddings)
                set_step(job_id, 'rag_db', 'done')

                tot_rag = len(docs)
                set_step(job_id, 'rag_embedding', 'active', details=f"Creating Embeddings [0/{tot_rag}]", progress=0, total=tot_rag)
                
                for i, doc in enumerate(docs):
                    vs.add_documents([doc])
                    set_step(job_id, 'rag_embedding', 'active', details=f"Creating Embeddings [{i+1}/{tot_rag}]", progress=i+1, total=tot_rag)
                
                set_step(job_id, 'rag_embedding', 'done')
                
                # Build the RAG Chain
                retriever = vector_store.get_retriever(vs, k=4)
                llm = get_llm(config)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert video AI assistant. Answer the user's question based ONLY on the video transcript context provided below.\n\nIf the answer is not found in the context, say: \"I could not find this information in the meeting transcript.\"\n\nAlways be concise and precise. If quoting someone, mention it clearly.\n\nContext from video transcript:\n{context}"),
                    ("human", "{question}"),
                ])
                rag_chain = ({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                
                with state_lock:
                    jobs[job_id]['rag_chain'] = rag_chain
                
                set_step(job_id, 'rag_complete', 'done', activity="RAG Ready")

            except Exception:
                traceback.print_exc()

        t1 = threading.Thread(target=do_summary, daemon=True)
        t2 = threading.Thread(target=do_rag, daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Extract action items / decisions / questions (can run after summary)
        with state_lock:
            jobs[job_id]['action_items'] = extract_action_items(full_transcript, config)
            jobs[job_id]['decisions'] = extract_key_decisions(full_transcript, config)
            jobs[job_id]['questions'] = extract_questions(full_transcript, config)
            jobs[job_id]['current_activity'] = "Completed"

    except Exception as e:
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        with state_lock:
            if job_id in jobs:
                jobs[job_id]['running'] = False
        
        # Cleanup temporary job files
        import shutil
        job_dir = os.path.join('downloads', job_id)
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
            except Exception as e:
                print(f"Failed to cleanup {job_dir}: {e}")


@app.route('/')
def index():
    return render_template('index.html')


import uuid

@app.route('/start', methods=['POST'])
def start():
    config = None
    if request.content_type and 'multipart/form-data' in request.content_type:
        language = request.form.get('language', 'english')
        source_type = request.form.get('source_type')
        config_str = request.form.get('config')
        if config_str:
            import json
            try:
                config = json.loads(config_str)
            except:
                pass
        if source_type == 'url':
            source = request.form.get('source')
        else:
            file = request.files.get('file')
            if not file or file.filename == '':
                return jsonify({'ok': False, 'error': 'No file selected'})
            
            job_id = str(uuid.uuid4())
            job_dir = os.path.join('downloads', job_id)
            os.makedirs(job_dir, exist_ok=True)
            filepath = os.path.join(job_dir, file.filename)
            file.save(filepath)
            source = filepath
    else:
        data = request.get_json() or {}
        source = data.get('source')
        language = data.get('language', 'english')
        config = data.get('config')

    if not config:
        return jsonify({'ok': False, 'error': 'Configuration payload is missing'})

    allow_local = os.environ.get('ALLOW_LOCAL_MODEL', 'true').lower() == 'true'
    if not allow_local:
        if config.get('transcription_mode') == 'offline' or config.get('embedding_mode') == 'offline':
            return jsonify({'ok': False, 'error': 'Local execution is disabled on this server. Please use online huggingface inference mode, or run the app locally on your own machine.'})

    if 'job_id' not in locals():
        job_id = str(uuid.uuid4())

    with state_lock:
        jobs[job_id] = get_initial_job_state(source, language, config)
        jobs[job_id]['running'] = True
        
    thread = threading.Thread(target=pipeline_worker, args=(job_id, source, language, config), daemon=True)
    thread.start()
    return jsonify({'ok': True, 'job_id': job_id})


@app.route('/status')
def status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({'ok': False, 'error': 'job_id missing'})
    with state_lock:
        if job_id not in jobs:
            return jsonify({'ok': False, 'error': 'Job not found'})
        job_state = jobs[job_id]
        out = {
            'running': job_state['running'],
            'source': job_state['source'],
            'language': job_state['language'],
            'transcript': bool(job_state['transcript']),
            'summary': job_state['summary'],
            'title': job_state['title'],
            'pipeline_steps': job_state['pipeline_steps'],
            'current_activity': job_state['current_activity']
        }
    return jsonify(out)

@app.route('/config', methods=['GET'])
def get_config_route():
    allow_local = os.environ.get('ALLOW_LOCAL_MODEL', 'true').lower() == 'true'
    repo = os.environ.get('GITHUB_REPO', 'https://github.com/your-username/video-agent')
    return jsonify({
        'allow_local_model': allow_local,
        'github_repo': repo
    })

@app.route('/config', methods=['POST'])
def set_config_route():
    data = request.get_json() or {}
    provider = data.get('provider')
    model = data.get('model')
    api_key = data.get('api_key')
    transcription_mode = data.get('transcription_mode')
    embedding_mode = data.get('embedding_mode')
    
    # Enforce local model restrictions
    allow_local = os.environ.get('ALLOW_LOCAL_MODEL', 'true').lower() == 'true'
    if not allow_local:
        if transcription_mode == 'offline' or embedding_mode == 'offline':
            return jsonify({'ok': False, 'error': 'Local execution is disabled on this server. Please use online huggingface inference mode, or run the app locally on your own machine.'})
            
    if provider and model and api_key and api_key != "********":
        from langchain.chat_models import init_chat_model
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
        
    return jsonify({'ok': True})


@app.route('/summary')
def get_summary():
    job_id = request.args.get('job_id')
    with state_lock:
        if job_id not in jobs: return jsonify({'ok': False})
        return jsonify({'summary': jobs[job_id]['summary']})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json() or {}
    q = data.get('question')
    job_id = data.get('job_id')
    if not q or not job_id:
        return jsonify({'ok': False, 'error': 'Question or job_id missing'})
    with state_lock:
        if job_id not in jobs:
            return jsonify({'ok': False, 'error': 'Job not found'})
        rag = jobs[job_id].get('rag_chain')
    if rag is None:
        return jsonify({'ok': False, 'error': 'RAG not ready'})
    try:
        answer = ask_question(rag, q)
        return jsonify({'ok': True, 'answer': answer})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


if __name__ == '__main__':
    app.run()

