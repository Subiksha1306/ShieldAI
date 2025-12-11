from pydantic import BaseModel
from typing import Optional

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    details: Optional[str] = None
