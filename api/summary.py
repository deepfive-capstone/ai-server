from fastapi import APIRouter
from pydantic import BaseModel
from services.summarizer import summarize_text

router = APIRouter()

class SummaryRequest(BaseModel):
    text: str

@router.post("/summarize")
def summarize(request: SummaryRequest):
    summary = summarize_text(request.text)
    return {
        "summary": summary
    }