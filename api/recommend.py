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
    query = make_recommend_query(req.title, req.category)
    recommendations = search_youtube_recommendations(query, req.limit)

    return {
        "query": query,
        "recommendations": recommendations
    }