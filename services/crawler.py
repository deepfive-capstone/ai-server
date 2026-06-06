import re
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


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
        "title": info.get("title") or "",
        "channel": info.get("channel") or "",
        "thumbnail": info.get("thumbnail") or ""
    }