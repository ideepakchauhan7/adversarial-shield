from pydantic import BaseModel, Field
from typing import Optional

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192, description="User prompt to classify")
    threshold: Optional[float] = Field(None, ge=0.5, le=1.0, description="Override default confidence threshold")

class ClassifyResponse(BaseModel):
    label:       str
    label_id:    int
    confidence:  float
    is_threat:   bool
    scores:      dict[str, float]
    latency_ms:  float
    explanation: str

class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)

class BatchClassifyResponse(BaseModel):
    results:       list[ClassifyResponse]
    total:         int
    threats_found: int
    total_ms:      float

class HealthResponse(BaseModel):
    status:      str
    model:       str
    device:      str
    version:     str = "1.0.0"