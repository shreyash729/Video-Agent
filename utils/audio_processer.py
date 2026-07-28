import yt_dlp
from pydub import AudioSegment
import os


def download_youtube_audio(url :str, job_id: str) ->str:
    job_dir = os.path.join('downloads', job_id)
    os.makedirs(job_dir, exist_ok=True)
    output_path = os.path.join(job_dir, "%(title)s.%(ext)s")
    browsers = ["chrome", "edge", "firefox", "brave", "opera", "safari"]
    filename = None
    import time
    import requests

    # First, attempt to use the third-party API
    try:
        print("Attempting to download via third-party API...")
        api_url = f"https://p.savenow.to/api/v2/download?format=480&url={url}&apikey=dfcb6d76f2f6a9894gjkege8a4ab232222"
        res = requests.get(api_url, timeout=10)
        res_json = res.json()
        if res_json.get("success"):
            progress_url = res_json.get("progress_url")
            if progress_url:
                max_retries = 30 # 60 seconds max
                for _ in range(max_retries):
                    time.sleep(2)
                    prog_res = requests.get(progress_url, timeout=10)
                    prog_json = prog_res.json()
                    if prog_json.get("success") == 1:
                        download_url = prog_json.get("download_url")
                        if download_url:
                            # Download the file
                            wav_resp = requests.get(download_url, stream=True, timeout=30)
                            wav_resp.raise_for_status()
                            out_filename = os.path.join(job_dir, "api_downloaded.mp4")
                            with open(out_filename, 'wb') as f:
                                for chunk in wav_resp.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print("Successfully downloaded via API.")
                            return convert_to_wav(out_filename)
                        break
    except Exception as e:
        print(f"Failed via third-party API: {e}")

    # Second, attempt without cookies (Works for most videos, and REQUIRED for headless servers)
    try:
        print("Attempting to download video...")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": output_path,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
    except Exception as e:
        print(f"Failed without cookies: {e}")
        # Fallback to cookies for local desktop usage if YouTube blocks the generic request
        for browser in browsers:
            try:
                print(f"Attempting to download with {browser} cookies...")
                ydl_opts = {
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "outtmpl": output_path,
                    "quiet": True,
                    "cookiesfrombrowser": (browser, ),
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                break
            except Exception as e2:
                print(f"Failed with {browser} cookies.")
                continue
            
    if not filename:
        raise Exception("Failed to download video. If it is age-restricted or YouTube is blocking the server IP, you may need to run the app locally where it can use your browser cookies.")
        
    return convert_to_wav(filename)



def convert_to_wav(input_path: str) ->  dict[str,str]:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) # 16 kHz (recommended for many ASR models)
    audio.export(output_path, format="wav")
    paths = {}
    paths["audio"] = output_path
    paths["video"] = input_path
    return paths




def process_input(source: str, job_id: str) -> dict[str,str]:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        paths = download_youtube_audio(source, job_id)
    else:
        print("Detected local file. Converting to WAV...")
        paths = convert_to_wav(source)
    '''
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    '''
    return paths

