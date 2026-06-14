from fastapi import APIRouter
from pydantic import BaseModel

from services.classifier import (
    predict_category,
    predict_category_with_score
)
from services.lora_summarizer import summarize_with_lora
from services.summarizer import improve_summary_with_gemini

from youtube_transcript_api import YouTubeTranscriptApi
import re
import yt_dlp
import time

router = APIRouter()

class AnalyzeRequest(BaseModel):
    url: str

def clean_transcript(text: str):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[a-zA-Z]{15,}', ' ', text)
    text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_video_id(url: str):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"shorts/([^?&]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

def get_youtube_transcript(video_id: str):
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=["ko", "en"])
    return " ".join([item.text for item in transcript])

def get_youtube_info(url: str):
    ydl_opts = {
        "quiet": True,
        "skip_download": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "title": info.get("title"),
        "thumbnail_url": info.get("thumbnail"),
        "channel": info.get("channel") or info.get("uploader")
    }

def pick_transcript_for_summary(text: str, max_chars: int = 800) -> str:
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    head_len = int(max_chars * 0.6)
    mid_len = int(max_chars * 0.25)
    tail_len = max_chars - head_len - mid_len

    head = text[:head_len]

    mid_start = max(0, len(text) // 2 - mid_len // 2)
    mid = text[mid_start:mid_start + mid_len]

    tail = text[-tail_len:]

    return (
        "[영상 앞부분]\n"
        f"{head}\n\n"
        "[영상 중간부분]\n"
        f"{mid}\n\n"
        "[영상 뒷부분]\n"
        f"{tail}"
    )

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    total_start = time.time()

    video_id = extract_video_id(request.url)

    if video_id is None:
        return {"error": "유효한 유튜브 URL이 아닙니다."}

    try:
        start = time.time()
        video_info = get_youtube_info(request.url)
        print(f"[analyze] 영상 정보 추출: {time.time() - start:.2f}초")
    except Exception:
        video_info = {
            "title": None,
            "thumbnail_url": None,
            "channel": None
        }

    try:
        start = time.time()
        transcript = get_youtube_transcript(video_id)
        print(f"[analyze] 자막 추출: {time.time() - start:.2f}초")

        start = time.time()
        transcript = clean_transcript(transcript)
        print(f"[analyze] 자막 전처리: {time.time() - start:.2f}초")

    except Exception as e:
        return {"error": "이 영상은 자막이 제공되지 않아 요약할 수 없습니다."}

    text_for_classification = (
        (video_info["title"] or "")
        + "\n"
        + transcript
    )

    start = time.time()
    result = predict_category_with_score(text_for_classification)
    print(f"[analyze] 카테고리 분류: {time.time() - start:.2f}초")

    category = result["category"]
    confidence = result["confidence"]

    try:
        '''
        start = time.time()
        qwen_summary = summarize_with_lora(transcript, category)
        print(f"[analyze] Qwen LoRA 요약: {time.time() - start:.2f}초")
        '''
        qwen_input = pick_transcript_for_summary(transcript, max_chars=800)
        gemini_input = pick_transcript_for_summary(transcript, max_chars=3000)
        
        print(f"[analyze] 원본 자막 길이: {len(transcript)}")
        print(f"[analyze] Qwen 입력 길이: {len(qwen_input)}")
        print(f"[analyze] Gemini 입력 길이: {len(gemini_input)}")

        start = time.time()
        qwen_summary = summarize_with_lora(qwen_input, category)
        print(f"[analyze] Qwen LoRA 요약: {time.time() - start:.2f}초")

        start = time.time()
        summary = improve_summary_with_gemini(
            #original_text=transcript,
            original_text=gemini_input,
            qwen_summary=qwen_summary,
            category=category
        )
        print(f"[analyze] Gemini 보정: {time.time() - start:.2f}초")

        summary_model = "qwen_lora + gemini_refine"

    except Exception as e:
        qwen_summary = None
        summary = f"요약 실패: {str(e)}"
        summary_model = "failed"

    print(f"[analyze] 전체 소요 시간: {time.time() - total_start:.2f}초")

    return {
        "video_id": video_id,
        "title": video_info["title"],
        "thumbnail_url": video_info["thumbnail_url"],
        "channel": video_info["channel"],
        "category": category,
        "confidence": confidence,
        "summary": summary,
    }