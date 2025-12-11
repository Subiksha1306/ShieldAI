from fastapi import APIRouter, UploadFile, File, Form
from app.schemas import PredictionResponse
from app.core.preprocessing_text import clean_text
from app.core.preprocessing_image import process_image
from app.core.lightweight_detectors import detect_text, detect_image

router = APIRouter(prefix="/multi", tags=["Multimodal Detection"])

@router.post("/", response_model=PredictionResponse)
async def analyze_multi(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    labels=[]
    confs=[]
    details=[]

    if text:
        t_label,t_conf,t_det = detect_text(clean_text(text))
        labels.append(t_label); confs.append(t_conf); details.append("text:"+t_det)

    if file:
        img_bytes = await file.read()
        img = process_image(img_bytes)
        i_label,i_conf,i_det = detect_image()
        labels.append(i_label); confs.append(i_conf); details.append("img:"+i_det)

    final = "THREAT" if "THREAT" in labels else "SAFE"
    avg_conf = round(sum(confs)/len(confs),2)

    return PredictionResponse(label=final,confidence=avg_conf,details=" | ".join(details))
