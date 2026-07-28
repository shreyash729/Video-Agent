import os
import tempfile
import time
import requests
from pydub import AudioSegment

# Auto-inject static ffmpeg and ffprobe binaries into PATH
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception as e:
    print(f"static_ffmpeg notification: {e}")

def get_job_dir(job_id: str) -> str:
    """Returns absolute path to a temporary job directory outside app root to avoid reloader restarts in production."""
    base_dir = os.getenv("DOWNLOAD_DIR", os.path.join(tempfile.gettempdir(), "video_agent_downloads"))
    job_dir = os.path.join(base_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir

def download_youtube_audio(url: str, job_id: str) -> dict[str, str]:
    job_dir = get_job_dir(job_id)
    
    print("Attempting to download via SaveNow API...")
    api_url = f"https://p.savenow.to/api/v2/download?format=480&url={url}&apikey=dfcb6d76f2f6a9894gjkege8a4ab232222"
    
    try:
        res = requests.get(api_url, timeout=10)
        res_json = res.json()
        if res_json.get("success"):
            progress_url = res_json.get("progress_url")
            if progress_url:
                max_retries = 30  # 60 seconds max
                for _ in range(max_retries):
                    time.sleep(2)
                    prog_res = requests.get(progress_url, timeout=10)
                    prog_json = prog_res.json()
                    if prog_json.get("success") == 1:
                        download_url = prog_json.get("download_url")
                        if download_url:
                            # Download the file stream
                            wav_resp = requests.get(download_url, stream=True, timeout=30)
                            wav_resp.raise_for_status()
                            out_filename = os.path.join(job_dir, "api_downloaded.mp4")
                            with open(out_filename, 'wb') as f:
                                for chunk in wav_resp.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print("Successfully downloaded via SaveNow API.")
                            return convert_to_wav(out_filename)
                        break
    except Exception as e:
        raise Exception(f"SaveNow API processing error: {e}")

    raise Exception("SaveNow API failed to process or download the requested YouTube URL.")


def convert_to_wav(input_path: str) -> dict[str, str]:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16 kHz (recommended for ASR models)
    audio.export(output_path, format="wav")
    paths = {}
    paths["audio"] = output_path
    paths["video"] = input_path
    return paths


def process_input(source: str, job_id: str) -> dict[str, str]:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        paths = download_youtube_audio(source, job_id)
    else:
        print("Detected local file. Converting to WAV...")
        paths = convert_to_wav(source)
    return paths
