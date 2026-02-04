"""
Omni-Vision Analytics API - Enterprise-grade FastAPI Application.

Features:
- Structured logging with request context
- Rate limiting and security headers
- Comprehensive error handling
- Prometheus metrics
- Health check with detailed system status
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Union, Dict
from contextlib import asynccontextmanager
from PIL import Image
import io
import sys
import time
import psutil
import pathlib

from .pipeline import OmniVisionPipeline
from .config import config
from .logging_config import setup_logging, get_api_logger, get_pipeline_logger
from .exceptions import (
    OmniVisionException,
    InvalidImageError,
    InvalidModeError,
    FileTooLargeError,
    UnsupportedFormatError
)
from .middleware import (
    RequestContextMiddleware,
    RateLimitMiddleware,
    LoggingMiddleware,
    SecurityHeadersMiddleware,
    ContentSizeLimitMiddleware
)
from .metrics import (
    registry,
    http_requests_total,
    http_request_duration_seconds,
    http_requests_active,
    update_gpu_metrics
)


# Setup logging
setup_logging(
    level=config.logging.level,
    format_type=config.logging.format_type,
    log_file=config.logging.log_file
)
logger = get_api_logger()


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Error detail for API responses."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: ErrorDetail


# =============================================================================
# Pydantic Schemas
# =============================================================================

class BoundingBox(BaseModel):
    """Represents a bounding box [x1, y1, x2, y2]."""
    coords: List[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")

    @validator("coords")
    def validate_coords(cls, v):
        if len(v) != 4:
            raise ValueError("coords must have exactly 4 values")
        if v[0] < 0 or v[1] < 0 or v[2] < v[0] or v[3] < v[1]:
            raise ValueError("invalid bounding box coordinates")
        return v


class Detection(BaseModel):
    """A single detected object."""
    label: str
    confidence: float = Field(..., ge=0, le=1)
    box: List[float]
    has_mask: Optional[bool] = None


class AnalysisMeta(BaseModel):
    """Metadata about the analysis process."""
    processing_mode: str
    objects_detected: int
    generated_queries: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    """Response model for the /analyze endpoint."""
    meta: AnalysisMeta
    detections: List[Detection]
    segmentation_available: bool
    masks_generated: Optional[int] = 0
    masks: Optional[List[Any]] = Field(default_factory=list, description="RLE dicts or base64 strings")
    mask_scores: Optional[List[float]] = Field(default_factory=list)
    mask_boxes: Optional[List[List[float]]] = Field(default_factory=list)
    mode_used: Optional[str] = "auto"
    mask_format: Optional[str] = "rle"
    queries_used: Optional[List[str]] = Field(default_factory=list)


class HealthStatus(str):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ModelHealth(BaseModel):
    """Health status of a specific model."""
    loaded: bool
    enabled: bool
    status: str


class SystemHealth(BaseModel):
    """System health information."""
    status: HealthStatus
    version: str
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    disk_usage_percent: float
    gpu_available: bool
    gpu_memory_mb: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: HealthStatus
    system: SystemHealth
    models: Dict[str, ModelHealth]
    details: Optional[Dict[str, str]] = None


# =============================================================================
# Lifecycle Management
# =============================================================================

pipeline: Optional[OmniVisionPipeline] = None
start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    global pipeline

    logger.info(
        "Initializing Omni-Vision Pipeline",
        extra={"extra_fields": {"environment": config.environment}}
    )

    try:
        pipeline = OmniVisionPipeline()
        await pipeline.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        # Continue anyway - individual model wrappers handle graceful degradation

    yield

    logger.info("Shutting down...")


# =============================================================================
# App Definition
# =============================================================================

app = FastAPI(
    title="Omni-Vision-Analytics",
    description="""
    Enterprise-grade computer vision pipeline for object detection and segmentation.

    **Features:**
    - Multi-model orchestration (YOLO, RF-DETR, Florence-2, SAM3)
    - Intelligent model selection based on scene complexity
    - Text-guided segmentation and smart query generation

    **Modes:**
    - `auto`: Detector-based flow → SAM3 box prompts
    - `query`: Text-prompted → fallback to detectors if empty
    - `smart_query`: Natural language → Gemma3 → SAM3
    """,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None
)


# =============================================================================
# Middleware Configuration
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RequestContextMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    ContentSizeLimitMiddleware,
    max_size_mb=config.api.max_request_size_mb
)

if config.api.rate_limit_enabled:
    app.add_middleware(
        RateLimitMiddleware,
        enabled=config.api.rate_limit_enabled
    )

if config.logging.log_requests:
    app.add_middleware(
        LoggingMiddleware,
        log_requests=config.logging.log_requests,
        log_body=config.logging.log_request_body
    )


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(OmniVisionException)
async def omni_vision_exception_handler(request: Request, exc: OmniVisionException) -> JSONResponse:
    """Handle custom OmniVisionExceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"{exc.code.value}: {exc.message}",
        extra={
            "extra_fields": {
                "request_id": request_id,
                "error_code": exc.code.value,
                "status_code": exc.status_code,
                "details": exc.details
            }
        }
    )

    http_requests_total.inc(
        method=request.method,
        endpoint=request.url.path,
        status=str(exc.status_code)
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code.value,
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard HTTPExceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"extra_fields": {"request_id": request_id, "status_code": exc.status_code}}
    )

    http_requests_total.inc(
        method=request.method,
        endpoint=request.url.path,
        status=str(exc.status_code)
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "request_id": request_id
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"extra_fields": {"request_id": request_id}}
    )

    http_requests_total.inc(
        method=request.method,
        endpoint=request.url.path,
        status="500"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "request_id": request_id
            }
        }
    )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """
    Comprehensive health check endpoint.

    Returns system status, model availability, and resource usage.
    """
    import torch

    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    # System metrics
    uptime = time.time() - start_time
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(pathlib.Path(__file__).anchor)
    cpu = psutil.cpu_percent(interval=0.1)

    # GPU info
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    # Determine overall health
    health_status = "healthy"
    issues = []

    if memory.percent > 90:
        health_status = "degraded"
        issues.append("high_memory_usage")

    if disk.percent > 90:
        health_status = "degraded"
        issues.append("high_disk_usage")

    if cpu > 90:
        health_status = "degraded"
        issues.append("high_cpu_usage")

    # Model health
    models = {}
    if pipeline:
        models["yolo"] = ModelHealth(
            loaded=pipeline.yolo.model is not None,
            enabled=True,
            status="ready" if pipeline.yolo.model is not None else "loading"
        )
        models["rf_detr"] = ModelHealth(
            loaded=pipeline.rf_detr.model is not None,
            enabled=True,
            status="ready" if pipeline.rf_detr.model is not None else "loading"
        )
        models["florence_2"] = ModelHealth(
            loaded=pipeline.florence_2.model is not None,
            enabled=True,
            status="ready" if pipeline.florence_2.model is not None else "loading"
        )
        models["sam_3"] = ModelHealth(
            loaded=pipeline.sam_3.enabled,
            enabled=pipeline.sam_3.enabled,
            status="ready" if pipeline.sam_3.enabled else "disabled"
        )
        models["gemma3"] = ModelHealth(
            loaded=pipeline.gemma3.enabled,
            enabled=pipeline.gemma3.enabled,
            status="ready" if pipeline.gemma3.enabled else "disabled"
        )

    system = SystemHealth(
        status=health_status,
        version="3.0.0",
        uptime_seconds=uptime,
        memory_usage_mb=memory.used / (1024 ** 2),
        cpu_percent=cpu,
        disk_usage_percent=disk.percent,
        gpu_available=gpu_available,
        gpu_memory_mb=gpu_memory
    )

    return HealthResponse(
        status=health_status,
        system=system,
        models=models,
        details={"issues": issues} if issues else None
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    update_gpu_metrics()
    return Response(
        content=registry.generate_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    user_text: Optional[str] = Form(None, description="Natural language input for smart query mode"),
    text_query: Optional[str] = Form(None, description="Direct text query for query mode"),
    mode: Optional[str] = Form(None, description="Inference mode: 'auto', 'query', or 'smart_query'"),
    mask_format: Optional[str] = Form("rle", description="Mask format: 'rle', 'png_base64', or 'none'")
):
    """
    Analyze an image using the intelligent model pipeline.

    **Modes:**
    - **auto** (default): YOLO → (RF-DETR if dense) → merge → SAM3 (box prompts)
    - **query**: SAM3 text-first → fallback to detectors if empty
    - **smart_query**: Gemma3 generates queries → SAM3 processes each

    **Mask formats:**
    - **rle** (default): Run-Length Encoded masks (COCO-compatible)
    - **png_base64**: Base64-encoded PNG images
    - **none**: Skip mask serialization (lightweight)

    **Examples:**
    - E-commerce cutout: `mode=auto`, no text_query
    - Privacy masking: `mode=query`, `text_query="face, license plate"`
    - Natural language: `mode=smart_query`, `user_text="この画像の中の車を見つけて"`
    """
    request_id = getattr(request.state, "request_id", "unknown")
    http_requests_active.inc()

    try:
        # Validate content type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise UnsupportedFormatError(
                content_type=file.content_type or "unknown",
                supported_formats=["image/jpeg", "image/png", "image/webp", "image/gif"]
            )

        # Validate file size
        content_length = request.headers.get("content-length")
        if content_length:
            size = int(content_length)
            max_size = config.api.max_image_size_mb * 1024 * 1024
            if size > max_size:
                raise FileTooLargeError(size, max_size)

        # Validate mode
        valid_modes = ["auto", "query", "smart_query"]
        if mode and mode not in valid_modes:
            raise InvalidModeError(mode, valid_modes)

        # Validate mask_format
        valid_formats = ["rle", "png_base64", "none"]
        if mask_format and mask_format not in valid_formats:
            raise InvalidModeError(mask_format, valid_formats)

        # Read and validate image
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise InvalidImageError("Empty file uploaded")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise InvalidImageError(f"Invalid image file: {str(e)}")

        # Run pipeline with tracking
        pipeline_logger = get_pipeline_logger(request_id)

        start_time = time.time()
        try:
            result = await pipeline.analyze(
                image=image,
                text_query=text_query,
                user_text=user_text,
                mode=mode,
                mask_format=mask_format or "rle"
            )
        finally:
            duration = time.time() - start_time

        pipeline_logger.info(
            "Pipeline completed",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "mode": result.get("mode_used", mode),
                    "objects_detected": result.get("meta", {}).get("objects_detected", 0),
                    "masks_generated": result.get("masks_generated", 0),
                    "duration_seconds": duration
                }
            }
        )

        # Record metrics
        http_requests_total.inc(
            method=request.method,
            endpoint=request.url.path,
            status="200"
        )
        http_request_duration_seconds.observe(
            duration,
            method=request.method,
            endpoint="/analyze"
        )

        return result

    except OmniVisionException:
        raise
    except Exception as e:
        logger.error(
            f"Analysis failed: {e}",
            exc_info=True,
            extra={"extra_fields": {"request_id": request_id}}
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during image analysis"
        )
    finally:
        http_requests_active.dec()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_config=None  # Use our custom logging
    )
