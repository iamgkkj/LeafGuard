"""
FastAPI application for LeafGuard AI.
Provides inference endpoints for plant disease classification.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import logging
from pathlib import Path
from typing import Optional, List
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch

from model_loader import ModelManager
from inference import predict, compare_models
from gradcam import generate_gradcam, overlay_heatmap, is_gradcam_available


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LeafGuard AI API",
    description="Plant disease classification using MobileNetV3-Small with CBAM attention",
    version="1.0.0"
)

# CORS configuration
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https?://.*",  # Avoid brittle mismatch on trailing slash/domain variants
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager: Optional[ModelManager] = None


# Pydantic models
class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    confidence: float
    top_predictions: List[dict]


class ComparisonResponse(BaseModel):
    baseline: PredictionResponse
    cbam: PredictionResponse
    agreement: bool


class HealthResponse(BaseModel):
    status: str
    baseline_loaded: bool
    cbam_loaded: bool
    device: str


class ClassesResponse(BaseModel):
    classes: List[str]
    num_classes: int


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global model_manager

    # Get checkpoint directory from environment or default
    # Default to runs/models relative to project root
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "runs/models"))

    # If relative path, resolve from project root (parent of backend directory)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(__file__).parent.parent / checkpoint_dir

    logger.info(f"Loading models from {checkpoint_dir}")

    model_manager = ModelManager(checkpoint_dir)
    load_status = model_manager.load_models()

    logger.info(f"Model loading status: {load_status}")
    
    if not load_status.get('baseline') and not load_status.get('cbam'):
        logger.error("Failed to load any models!")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check application health and model loading status."""
    if model_manager is None:
        return HealthResponse(
            status="initializing",
            baseline_loaded=False,
            cbam_loaded=False,
            device="unknown"
        )
    
    return HealthResponse(
        status="ready",
        baseline_loaded=model_manager.baseline_model is not None,
        cbam_loaded=model_manager.cbam_model is not None,
        device=str(model_manager.device)
    )


@app.get("/v1/models")
async def openai_compatible_models():
    """Minimal OpenAI-compatible model listing for clients/probes that expect it."""
    return {
        "object": "list",
        "data": [
            {"id": "leafguard-baseline", "object": "model", "owned_by": "leafguard"},
            {"id": "leafguard-cbam", "object": "model", "owned_by": "leafguard"},
        ],
    }


@app.get("/classes", response_model=ClassesResponse)
async def get_classes():
    """Get the list of supported disease classes."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    classes = model_manager.get_classes()
    return ClassesResponse(
        classes=classes,
        num_classes=len(classes)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    model: str = Form(...)
):
    """
    Predict disease class for an uploaded image.
    
    Args:
        file: Uploaded image file
        model: Model to use ('baseline' or 'cbam')
        
    Returns:
        Prediction results with confidence and top-k predictions
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    # Validate model selection
    if model not in ['baseline', 'cbam']:
        raise HTTPException(status_code=400, detail="Model must be 'baseline' or 'cbam'")
    
    # Check if model is loaded
    try:
        model_obj = model_manager.get_model(model)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Validate image size (reasonable limit)
        if max(image.size) > 10000:
            raise HTTPException(status_code=400, detail="Image too large")
        
        # Run prediction
        result = predict(
            model_obj,
            image,
            model_manager.get_classes(),
            model_manager.device
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/compare", response_model=ComparisonResponse)
async def compare_endpoint(file: UploadFile = File(...)):
    """
    Run both models on the same image and compare results.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Comparison results from both models
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    # Check if both models are loaded
    if model_manager.baseline_model is None or model_manager.cbam_model is None:
        raise HTTPException(status_code=503, detail="Both models must be loaded for comparison")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        if max(image.size) > 10000:
            raise HTTPException(status_code=400, detail="Image too large")
        
        # Run comparison
        result = compare_models(
            model_manager.baseline_model,
            model_manager.cbam_model,
            image,
            model_manager.get_classes(),
            model_manager.device
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...),
    model: str = Form(...)
):
    """
    Generate Grad-CAM explanation for a prediction.
    
    Args:
        file: Uploaded image file
        model: Model to use ('baseline' or 'cbam')
        
    Returns:
        JSON with heatmap data (base64 encoded) or error if unavailable
    """
    if not is_gradcam_available():
        raise HTTPException(status_code=501, detail="Grad-CAM not available (pytorch-grad-cam not installed)")
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    # Validate model selection
    if model not in ['baseline', 'cbam']:
        raise HTTPException(status_code=400, detail="Model must be 'baseline' or 'cbam'")
    
    try:
        model_obj = model_manager.get_model(model)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        if max(image.size) > 10000:
            raise HTTPException(status_code=400, detail="Image too large")
        
        # First get prediction to know the target label
        prediction = predict(
            model_obj,
            image,
            model_manager.get_classes(),
            model_manager.device
        )
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(
            model_obj,
            image,
            prediction['predicted_label'],
            model_manager.device
        )
        
        if heatmap is None:
            raise HTTPException(status_code=500, detail="Failed to generate Grad-CAM heatmap")
        
        # Create overlay
        overlay = overlay_heatmap(image, heatmap)
        
        # Convert to base64 for JSON response
        import base64
        buffered = BytesIO()
        overlay.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "prediction": prediction,
            "heatmap_image": f"data:image/png;base64,{img_str}"
        }
    
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")


@app.get("/history")
async def get_history():
    """
    Get recent prediction history.
    
    Note: History feature is optional and not implemented in this minimal version.
    Returns empty list.
    """
    return {"history": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
