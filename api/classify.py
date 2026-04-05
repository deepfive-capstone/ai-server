from fastapi import APIRouter
from pydantic import BaseModel
from services.classifier import predict_category

router = APIRouter()

class ClassifyRequest(BaseModel):
    text: str

@router.post("/classify")
def classify(request: ClassifyRequest):
    category = predict_category(request.text)
    return {"category": category}
