from fastapi import APIRouter
from pydantic import BaseModel

from services.classifier import predict_category_with_score
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
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[a-zA-Z]{15,}", " ", text)
    text = re.sub(r"(\b\w+\b)( \1\b)+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
        "title": info.get("title") or "",
        "thumbnail_url": info.get("thumbnail") or "",
        "channel": info.get("channel") or info.get("uploader") or ""
    }


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

    except Exception as e:
        print(f"[analyze] 영상 정보 추출 실패: {e}")

        video_info = {
            "title": "",
            "thumbnail_url": "",
            "channel": ""
        }

    try:
        start = time.time()
        transcript = get_youtube_transcript(video_id)
        print(f"[analyze] 자막 추출: {time.time() - start:.2f}초")

        start = time.time()
        transcript = clean_transcript(transcript)
        print(f"[analyze] 자막 전처리: {time.time() - start:.2f}초")

    except Exception as e:
        print(f"[analyze] 자막 추출 실패: {e}")

        return {
            "video_id": video_id,
            "title": video_info["title"],
            "thumbnail_url": video_info["thumbnail_url"],
            "channel": video_info["channel"],
            "error": "이 영상은 자막이 제공되지 않아 요약할 수 없습니다."
        }

    # classify-youtube와 동일한 분류 입력 구조
    text_for_classification = (
        (video_info["title"] or "")
        + "\n채널: "
        + (video_info["channel"] or "")
        + "\n"
        + transcript
    )

    start = time.time()
    result = predict_category_with_score(
        text_for_classification,
        title=video_info["title"]
    )
    print(f"[analyze] 카테고리 분류: {time.time() - start:.2f}초")

    category = result["category"]
    confidence = result["confidence"]

    try:
        start = time.time()
        qwen_summary = summarize_with_lora(transcript, category)
        print(f"[analyze] Qwen LoRA 요약: {time.time() - start:.2f}초")

        start = time.time()
        summary = improve_summary_with_gemini(
            original_text=transcript,
            qwen_summary=qwen_summary,
            category=category
        )
        print(f"[analyze] Gemini 보정: {time.time() - start:.2f}초")

    except Exception as e:
        print(f"[analyze] 요약 실패: {e}")
        summary = f"요약 실패: {str(e)}"

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