from fastapi import FastAPI
from api.recommend import router as recommend_router
from api.summary import router as summary_router
from api.classify import router as classify_router
from api.analyze import router as analyze_router
from services.summarizer import summarize_text
from services.crawler import (
    extract_video_id,
    get_youtube_transcript,
    get_video_info
)

app = FastAPI()
app.include_router(summary_router)
app.include_router(classify_router)
app.include_router(analyze_router)
app.include_router(recommend_router)

@app.get("/")
def root():
    return {"message": "AI server is running"}


@app.get("/transcript")
def get_transcript(url: str):
    video_id = extract_video_id(url)

    if not video_id:
        return {"error": "유효한 유튜브 링크가 아닙니다."}

    try:
        transcript_text = get_youtube_transcript(video_id)
        video_info = get_video_info(video_id)

        return {
            "video_id": video_id,
            "title": video_info["title"],
            "channel": video_info["channel"],
            "thumbnail": video_info["thumbnail"],
            "transcript": transcript_text
        }
    except Exception as e:
        return {
            "video_id": video_id,
            "error": str(e)
        }
    
@app.get("/youtube-summary")
def youtube_summary(url: str):
    video_id = extract_video_id(url)

    if not video_id:
        return {"error": "유효한 유튜브 링크가 아닙니다."}

    try:
        transcript_text = get_youtube_transcript(video_id)
        video_info = get_video_info(video_id)
        summary = summarize_text(transcript_text)

        return {
            "video_id": video_id,
            "title": video_info["title"],
            "channel": video_info["channel"],
            "thumbnail": video_info["thumbnail"],
            "summary": summary
        }
    except Exception as e:
        return {
            "video_id": video_id,
            "error": str(e)
        }