import json
import math
import mimetypes
from pathlib import Path
from typing import Dict, Any

# Media processing libraries for token estimation
import cv2
from mutagen import File as MutagenFile
from PIL import Image
from google import genai
from .uno_bench_loader import UNOBenchLoader

# Assuming UNOBenchLoader and UNOBenchSample are defined above this script

# --- ADDED: The Evaluation Instruction Suffix ---
EVAL_INSTRUCTION_SUFFIX = """

**FINAL OUTPUT INSTRUCTIONS:**
You will use your internal reasoning process to analyze the inputs first. When generating your final response, you must adhere strictly to the following formatting rules:

* **For multiple-choice questions:** Your final output must contain **ONLY** the exact letter or number of the correct choice (e.g., A, B, C, 1, 2). **Do not** include any rationale, restatement of the answer, markdown formatting, or disclaimers in the final output. 
* **For open-ended questions:** Provide only the direct, concise answer. **Do not** include introductory phrases, rationale, or disclaimers."""


def get_mime_type(file_path: str | Path) -> str:
    """Extracts the file extension and guesses the MIME type dynamically."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"

def estimate_image_tokens(image_path: str | Path) -> int:
    """Calculates image tokens based on dimensions and 768x768 tiling."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if w <= 384 and h <= 384:
                return 258
            else:
                tiles_w = math.ceil(w / 768)
                tiles_h = math.ceil(h / 768)
                return tiles_w * tiles_h * 258
    except Exception as e:
        print(f"Warning: Could not read image {image_path} for token estimation: {e}")
        return 258  # Fallback to base cost

def estimate_audio_tokens(audio_path: str | Path) -> int:
    """Calculates audio tokens based on duration (32 tokens/sec)."""
    try:
        audio = MutagenFile(str(audio_path))
        if audio is not None and audio.info is not None:
            duration_seconds = audio.info.length
            return math.ceil(duration_seconds * 32)
        return 0
    except Exception as e:
        print(f"Warning: Could not read audio {audio_path} for token estimation: {e}")
        return 0

def estimate_video_tokens(video_path: str | Path) -> int:
    """Calculates video tokens based on duration (263 tokens/sec)."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0:
            duration_seconds = frame_count / fps
            return math.ceil(duration_seconds * 263)
        return 0
    except Exception as e:
        print(f"Warning: Could not read video {video_path} for token estimation: {e}")
        return 0

def upload_and_get_part(
    file_path: str | Path, 
    client: genai.Client, 
    cache: Dict[str, str]
) -> Dict[str, Any]:
    """Uploads to File API (if not cached) and returns the JSON structure."""
    file_path_str = str(file_path)
    mime_type = get_mime_type(file_path)

    if file_path_str in cache:
        file_uri = cache[file_path_str]
    else:
        print(f"Uploading: {Path(file_path).name}...")
        try:
            uploaded_file = client.files.upload(
                file=file_path_str,
                config={'mime_type': mime_type}
            )
            file_uri = uploaded_file.uri
            cache[file_path_str] = file_uri
        except Exception as e:
            print(f"Error uploading {file_path_str}: {e}")
            return {}

    return {
        "fileData": {
            "mimeType": mime_type,
            "fileUri": file_uri
        }
    }

def build_jsonl_batch_with_estimation(
    loader: 'UNOBenchLoader', 
    client: genai.Client,
    output_filepath: str = "unobench_batch_file_api.jsonl",
    thinking_budget: int = 4096,
    temperature: float = 1.0,
    n_candidates: int = 50,
    include_thoughts: bool = True
):
    """Builds the JSONL and estimates token usage across the entire dataset."""
    media_cache: Dict[str, str] = {}
    
    # Token Tracking Counters
    total_text_tokens = 0
    total_image_tokens = 0
    total_audio_tokens = 0
    total_video_tokens = 0
    
    with open(output_filepath, 'w', encoding='utf-8') as jsonl_file:
        for idx, sample in enumerate(loader):
            parts = []
            print(f"Processing sample {idx + 1}/{len(loader)} (ID: {sample.id})")
            
            # 1. Text (Question) --- UPDATED WITH SUFFIX ---
            if sample.question:
                full_prompt_text = sample.question + EVAL_INSTRUCTION_SUFFIX
                parts.append({"text": full_prompt_text})
                # Update text token estimation based on the combined length
                total_text_tokens += math.ceil(len(full_prompt_text) / 4.0)
            
            # 2. Images
            if sample.image_paths:
                for img_path in sample.image_paths:
                    part = upload_and_get_part(img_path, client, media_cache)
                    if part: 
                        parts.append(part)
                        total_image_tokens += estimate_image_tokens(img_path)
                    
            # 3. Audio
            if sample.audio_paths:
                for audio_path in sample.audio_paths:
                    part = upload_and_get_part(audio_path, client, media_cache)
                    if part: 
                        parts.append(part)
                        total_audio_tokens += estimate_audio_tokens(audio_path)
                    
            # 4. Video
            if sample.video_paths:
                for video_path in sample.video_paths:
                    part = upload_and_get_part(video_path, client, media_cache)
                    if part: 
                        parts.append(part)
                        total_video_tokens += estimate_video_tokens(video_path)
            
            # 5. Construct the CORRECT Batch API request dictionary
            request_dict = {
                # The 'key' tracks the request. When the batch is done, 
                # the output will include this exact key so you can match them up.
                "key": str(sample.id), 
                
                # The 'request' wrapper holds the actual payload
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": parts
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "candidateCount": n_candidates,
                        "thinkingConfig": {
                            "thinkingBudget": thinking_budget,
                            "includeThoughts": include_thoughts
                        }
                    }
                }
            }
            
            # Write the single JSON line
            jsonl_file.write(json.dumps(request_dict) + '\n')
            
    # Calculate Grand Total
    grand_total_tokens = sum([
        total_text_tokens, 
        total_image_tokens, 
        total_audio_tokens, 
        total_video_tokens
    ])

    print("\n" + "="*40)
    print("      BATCH PROCESSING COMPLETE      ")
    print("="*40)
    print(f"Total Unique Media Uploads: {len(media_cache)}")
    print(f"Output File: {output_filepath}\n")
    print("--- ESTIMATED INPUT TOKENS ---")
    print(f"Text:   {total_text_tokens:,}")
    print(f"Images: {total_image_tokens:,}")
    print(f"Audio:  {total_audio_tokens:,}")
    print(f"Video:  {total_video_tokens:,}")
    print("-" * 30)
    print(f"GRAND TOTAL: {grand_total_tokens:,} tokens")
    print("="*40)

# --- Execution Example ---
if __name__ == "__main__":
    client = genai.Client()
    loader = UNOBenchLoader(data_path="/scratch/datasets/uno-bench", split="validation")
    
    build_jsonl_batch_with_estimation(
        loader=loader, 
        client=client,
        output_filepath="unobench_final_requests.jsonl"
    )
