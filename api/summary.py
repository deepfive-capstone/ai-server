from fastapi import APIRouter
from pydantic import BaseModel
from services.summarizer import summarize_text

router = APIRouter()

class SummaryRequest(BaseModel):
    text: str

@router.post("/summary")
def summary(request: SummaryRequest):
    summary_result = summarize_text(request.text)
    return {"summary": summary_result}
