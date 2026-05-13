from pydantic import BaseModel
from fastapi import APIRouter
from services.summarizer import summarize_text

router = APIRouter()

class SummaryRequest(BaseModel):
    text: str
    category: str = "기타"

@router.post("/summarize")
def summarize(request: SummaryRequest):
    summary = summarize_text(request.text, request.category)
    return {
        "category": request.category,
        "summary": summary
    }