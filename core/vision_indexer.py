import os
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from core.config import get_vision_model

from utils.audio_processer import get_job_dir

def extract_and_caption(
    video_path: str, 
    job_id: str, 
    interval_sec: int = 5, 
    similarity_threshold: float = 0.75
) -> list[dict]:
    # Ensure save directory exists outside app root to avoid server reloader restarts
    save_dir = get_job_dir(job_id)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Fallback guard
        
    frame_interval = int(fps * interval_sec)
    if frame_interval <= 0:
        frame_interval = 1
        
    visual_captions = []
    prev_gray = None
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp_sec = frame_count / fps
            
            is_unique = True
            if prev_gray is not None:
                try:
                    score, _ = ssim(prev_gray, gray, full=True)
                    if score >= similarity_threshold:
                        is_unique = False  # Skip redundant frame
                except Exception:
                    is_unique = True
            
            if is_unique:
                prev_gray = gray
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Format timestamp string cleanly (MM:SS)
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                image_path = os.path.join(save_dir, f"frame_{timestamp_sec:.2f}.png")
                pil_img.save(image_path)
                
                visual_captions.append({
                    "timestamp": time_str,
                    "image_path": image_path
                })
                
        frame_count += 1
        
    cap.release()
    return visual_captions


def process_vision_captions(video_path: str, job_id: str, config: dict):
    """
    Helper function to run extraction and query the selected Vision Model.
    """
    vision_model = get_vision_model(config)
    if not vision_model:
        return False  # Vision was toggled off by user
        
    visual_frames = extract_and_caption(video_path, job_id=job_id)
    if not visual_frames:
        return []
        
    # Analyze extracted frames with the vision model
    captions = vision_model.analyze(
        visual_frames, 
        prompt="Describe this image scene in details"
    )
    return captions