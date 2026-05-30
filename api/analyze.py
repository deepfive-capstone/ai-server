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

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    video_id = extract_video_id(request.url)

    if video_id is None:
        return {"error": "유효한 유튜브 URL이 아닙니다."}

    try:
        video_info = get_youtube_info(request.url)
    except Exception:
        video_info = {
            "title": None,
            "thumbnail_url": None,
            "channel": None
        }

    try:
        transcript = get_youtube_transcript(video_id)
        transcript = clean_transcript(transcript)
    except Exception as e:
        return {"error": f"자막 추출 실패: {str(e)}"}

    text_for_classification = (
        (video_info["title"] or "")
        + "\n"
        + transcript
    )

    result = predict_category_with_score(
         text_for_classification
    )

    category = result["category"]
    confidence = result["confidence"]

    try:
        qwen_summary = summarize_with_lora(transcript, category)

        summary = improve_summary_with_gemini(
            original_text=transcript,
            qwen_summary=qwen_summary,
            category=category
        )

        summary_model = "qwen_lora + gemini_refine"

    except Exception as e:
        qwen_summary = None
        summary = f"요약 실패: {str(e)}"
        summary_model = "failed"

    return {
        "video_id": video_id,
        "title": video_info["title"],
        "thumbnail_url": video_info["thumbnail_url"],
        "channel": video_info["channel"],
        "category": category,
        "confidence": confidence,
        "summary": summary,
    }