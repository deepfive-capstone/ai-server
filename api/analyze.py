from fastapi import APIRouter
from pydantic import BaseModel

from services.classifier import predict_category
from services.summarizer import summarize_text
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

    category = predict_category(transcript)
    summary = summarize_text(transcript, category)

    return {
        "video_id": video_id,
        "title": video_info["title"],
        "thumbnail_url": video_info["thumbnail_url"],
        "channel": video_info["channel"],
        "category": category,
        "summary": summary
    }