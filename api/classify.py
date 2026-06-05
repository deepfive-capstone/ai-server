from fastapi import APIRouter
from pydantic import BaseModel
from services.classifier import (
	predict_category,
	predict_category_with_score
)
from services.crawler import (
    extract_video_id,
    get_youtube_transcript,
    get_video_info
)

router = APIRouter()

class ClassifyRequest(BaseModel):
    text: str

@router.post("/classify")
def classify(request: ClassifyRequest):
    category = predict_category(request.text)
    return {"category": category}

class YoutubeClassifyRequest(BaseModel):
    url: str

@router.post("/classify-youtube")
def classify_youtube(request: YoutubeClassifyRequest):

    video_id = extract_video_id(request.url)

    if not video_id:
        return {"error": "유효한 유튜브 링크가 아닙니다."}

    video_info = get_video_info(video_id)

    title = video_info["title"]
    channel = video_info["channel"]
    thumbnail = video_info["thumbnail"]

    transcript = get_youtube_transcript(video_id)

    text = (
        title + "\n" +
        title + "\n" +
        title + "\n" +
        transcript[:1000]
    )


    result = predict_category_with_score(text)

    category = result["category"]
    confidence = result["confidence"]

    return {
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "thumbnail": thumbnail,
        "category": category,
        "confidence": confidence
    }