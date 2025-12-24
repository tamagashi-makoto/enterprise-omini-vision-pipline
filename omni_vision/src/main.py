"""
Omni-Vision Analytics API - FastAPI Application
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
from PIL import Image
import io

from .pipeline import OmniVisionPipeline

# --- Pydantic Schemas ---

class BoundingBox(BaseModel):
    """Represents a bounding box [x1, y1, x2, y2]."""
    coords: List[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")


class Detection(BaseModel):
    """A single detected object."""
    label: str
    confidence: float
    box: List[float]
    has_mask: Optional[bool] = None


class AnalysisMeta(BaseModel):
    """Metadata about the analysis process."""
    processing_mode: str
    objects_detected: int


class AnalysisResponse(BaseModel):
    """Response model for the /analyze endpoint."""
    meta: AnalysisMeta
    detections: List[Detection]
    segmentation_available: bool
    masks_generated: Optional[int] = 0


# --- Lifecycle Management ---

pipeline = OmniVisionPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Initializing Omni-Vision Pipeline...")
    await pipeline.load_models()
    yield
    print("Shutting down...")


# --- App Definition ---

app = FastAPI(
    title="Omni-Vision-Analytics",
    description="Intelligent orchestration pipeline for computer vision model selection.",
    version="2.0.0",
    lifespan=lifespan
)

# --- Endpoints ---

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint to verify system status."""
    import torch
    return {
        "status": "healthy", 
        "models_loaded": pipeline.models_loaded,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    text_query: Optional[str] = Form(None, description="Optional text query for Florence-2")
):
    """
    Analyze an image using the intelligent model pipeline.
    
    - **Stage 1**: YOLOv12 for screening.
    - **Stage 2**: Switch to RF-DETR if density > 15.
    - **Stage 3**: Use Florence-2 if `text_query` is provided.
    - **Stage 4**: Generate segmentation masks with SAM 3.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read image bytes and convert to PIL Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run the pipeline
        result = await pipeline.analyze(image=image, text_query=text_query)
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
