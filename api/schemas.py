from typing import List, Optional
from pydantic import BaseModel


class TopPrediction(BaseModel):
    label: str
    confidence: float
    class_idx: int


class PredictResponse(BaseModel):
    predicted_class: str
    class_idx: int
    confidence: float
    top3: List[TopPrediction]


class GradCAMResponse(BaseModel):
    predicted_class: str
    class_idx: int
    confidence: float
    method: str
    overlay_base64: str  # PNG base64
    heatmap_base64: str


class ModelMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    params: int
    flops: int
    size_mb: float
    fps_cpu: float
    fps_gpu: Optional[float] = None
    per_class_f1: Optional[List[float]] = None


class CompareResponse(BaseModel):
    cbam: ModelMetrics
    baseline: ModelMetrics
    accuracy_gain_pct: float
    f1_gain_pct: float


class ClassInfo(BaseModel):
    idx: int
    name: str
    crop: str
    disease: str
    description: str


class HistoryEntry(BaseModel):
    id: Optional[int] = None
    timestamp: str
    predicted_class: str
    class_idx: int
    confidence: float
    top3: List[TopPrediction]
    image_base64: Optional[str] = None  # thumbnail
    gradcam_base64: Optional[str] = None


class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]
    total: int
