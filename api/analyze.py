from fastapi import APIRouter
from pydantic import BaseModel

from services.classifier import predict_category
from services.summarizer import summarize_text
from youtube_transcript_api import YouTubeTranscriptApi
import re

router = APIRouter()

class AnalyzeRequest(BaseModel):
    url: str

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

    full_text = " ".join([item.text for item in transcript])
    return full_text

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    video_id = extract_video_id(request.url)

    if video_id is None:
        return {"error": "유효한 유튜브 URL이 아닙니다."}

    try:
        transcript = get_youtube_transcript(video_id)
    except Exception as e:
        return {"error": f"자막 추출 실패: {str(e)}"}

    category = predict_category(transcript)
    summary = summarize_text(transcript)

    return {
        "video_id": video_id,
        "category": category,
        "summary": summary
    }