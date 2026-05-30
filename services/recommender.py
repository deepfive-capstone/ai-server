import yt_dlp
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")


def make_recommend_query(title: str, category: str = "", keywords: list[str] = None):
    if keywords is None:
        keywords = []

    keyword_text = " ".join(keywords)
    return f"{category} {keyword_text} {title}".strip()


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_youtube_candidates(query: str, max_results: int = 20):
    search_url = f"ytsearch{max_results}:{query}"

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=False)

    candidates = []

    for rank, item in enumerate(info.get("entries", []), start=1):
        video_id = item.get("id")
        title = item.get("title") or ""
        channel = item.get("uploader") or ""

        if not video_id or not title:
            continue

        candidates.append({
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "thumbnail_url": item.get("thumbnail"),
            "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
            "search_rank": rank
        })

    return candidates


def rerank_by_embedding(
    input_title: str,
    category: str,
    summary: str,
    candidates: list[dict],
    top_k: int = 5
):
    input_text = f"""
    카테고리: {category}
    제목: {input_title}
    요약: {summary}
    """

    input_embedding = embedding_model.encode(input_text)

    reranked = []

    for candidate in candidates:
        candidate_text = f"""
        제목: {candidate["title"]}
        채널: {candidate["channel"]}
        """

        candidate_embedding = embedding_model.encode(candidate_text)

        semantic_score = cosine_similarity(input_embedding, candidate_embedding)

        rank_score = 1 / candidate["search_rank"]

        final_score = (0.85 * semantic_score) + (0.15 * rank_score)

        reranked.append({
            **candidate,
            "semantic_score": round(semantic_score, 4),
            "final_score": round(final_score, 4)
        })

    reranked.sort(key=lambda x: x["final_score"], reverse=True)

    return reranked[:top_k]


def recommend_videos(
    title: str,
    category: str = "",
    summary: str = "",
    keywords: list[str] = None,
    limit: int = 5,
    candidate_count: int = 20
):
    if keywords is None:
        keywords = []

    query = make_recommend_query(title, category, keywords)

    candidates = search_youtube_candidates(
        query=query,
        max_results=candidate_count
    )

    recommendations = rerank_by_embedding(
        input_title=title,
        category=category,
        summary=summary,
        candidates=candidates,
        top_k=limit
    )

    return {
        "recommend_type": "hybrid_search_embedding_reranking",
        "query": query,
        "candidate_count": len(candidates),
        "recommendations": recommendations
    }

def search_youtube_recommendations(
    title: str,
    category: str = "",
    summary: str = "",
    keywords: list[str] = None,
    limit: int = 5,
    candidate_count: int = 20
):
    return recommend_videos(
        title=title,
        category=category,
        summary=summary,
        keywords=keywords,
        limit=limit,
        candidate_count=candidate_count
    )