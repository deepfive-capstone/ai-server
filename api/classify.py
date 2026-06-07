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

    title = video_info.get("title") or ""
    channel = video_info.get("channel") or ""
    thumbnail = video_info.get("thumbnail") or ""
    
    try:
        transcript = get_youtube_transcript(video_id)
    except Exception as e:
        print(f"[classify-youtube] 자막 추출 실패: {e}")
        transcript = ""

    text = (
        title + "\n"
        + "채널: " + channel + "\n"
        + transcript
    )

    print("===== 서비스 분류 입력 확인 =====")
    print("title:", title)
    print("channel:", channel)
    print("transcript length:", len(transcript))
    print("text preview:")
    print(text[:1500])
    print("==============================")


    result = predict_category_with_score(text, title=title)

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