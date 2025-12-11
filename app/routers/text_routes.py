from fastapi import APIRouter
from app.schemas import TextRequest, PredictionResponse
from app.core.preprocessing_text import clean_text
from app.core.lightweight_detectors import detect_text

router = APIRouter(prefix="/text", tags=["Text Detection"])

@router.post("/", response_model=PredictionResponse)
def analyze_text(data: TextRequest):
    text = clean_text(data.text)
    label, conf, details = detect_text(text)
    return PredictionResponse(label=label,confidence=conf,details=details)
