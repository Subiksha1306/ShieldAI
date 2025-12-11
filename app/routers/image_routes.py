from fastapi import APIRouter, UploadFile, File
from app.schemas import PredictionResponse
from app.core.preprocessing_image import process_image
from app.core.lightweight_detectors import detect_image

router = APIRouter(prefix="/image", tags=["Image Detection"])

@router.post("/", response_model=PredictionResponse)
async def analyze_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = process_image(img_bytes)  # processed but unused now
    label, conf, details = detect_image()
    return PredictionResponse(label=label,confidence=conf,details=details)
