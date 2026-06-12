from fastapi import APIRouter
from pydantic import BaseModel
from services.recommender import make_recommend_query, search_youtube_recommendations

router = APIRouter(prefix="/recommend", tags=["recommend"])


class RecommendRequest(BaseModel):
    title: str
    category: str = ""
    limit: int = 10


@router.post("")
def recommend_videos(req: RecommendRequest):
    recommendations = search_youtube_recommendations(
        title=req.title,
        category=req.category,
        limit=req.limit
    )

    return {
        "query": recommendations.get("query", ""),
        "recommendations": recommendations
    }