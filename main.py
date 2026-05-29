from fastapi import FastAPI
from youtube_transcript_api import YouTubeTranscriptApi
from api.recommend import router as recommend_router
from api.summary import router as summary_router
from api.classify import router as classify_router
from api.analyze import router as analyze_router
from api.recommend import router as recommend_router
import re
import yt_dlp
from services.summarizer import summarize_text

app = FastAPI()
app.include_router(summary_router)
app.include_router(classify_router)
app.include_router(analyze_router)
app.include_router(recommend_router)


def extract_video_id(url: str):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_id: str):
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["ko", "en"])
    full_text = " ".join([item.text for item in transcript])
    return full_text


def get_video_info(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "title": info.get("title"),
        "channel": info.get("channel"),
        "thumbnail": info.get("thumbnail")
    }


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