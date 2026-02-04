# Omni-Vision Analytics

Computer vision pipeline for object detection and segmentation. Multi-model orchestration with SAM3 for pixel-level masks.

## Quick Start

```bash
# Background removal
curl -X POST http://localhost:8000/analyze -F "file=@product.jpg"

# Privacy masking
curl -X POST http://localhost:8000/analyze -F "file=@photo.jpg" -F "text_query=face" -F "mode=query"

# Defect detection
curl -X POST http://localhost:8000/analyze -F "file=@pcb.jpg" -F "text_query=scratch, crack" -F "mode=query"
```

## Architecture

```
Input -> YOLOv12 -> RF-DETR (if dense) -> NMS -> Filter -> SAM3 -> Masks
```

**Models**: YOLOv12 (fast), RF-DETR (dense), Florence-2 (open-vocab), SAM3 (segmentation)

**Modes**:
- `auto`: YOLO first, RF-DETR if needed
- `query`: Text prompt, fallback to detectors
- `smart_query`: Natural language via Gemma3

## Setup

Docker:
```bash
docker-compose up --build
```

Local:
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
python -m uvicorn src.main:app
```

## API

### POST /analyze

| Param | Type | Description |
|-------|------|-------------|
| file | file | Image (required) |
| text_query | string | Search query |
| mode | string | auto/query/smart_query |
| mask_format | string | rle/png_base64/none |

### GET /health

System status and model availability.

### GET /metrics

Prometheus metrics.

## Config

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| LOG_LEVEL | INFO | Logging level |
| RATE_LIMIT_PER_MINUTE | 60 | Rate limit |
| MAX_IMAGE_SIZE_MB | 50 | Max upload size |

## Testing

```bash
pytest tests/test_pipeline_unit.py -v
pytest tests/test_api.py -v
RUN_INTEGRATION_TESTS=1 pytest -m integration
```

## File Structure

```
src/
├── main.py           # API entry point
├── pipeline.py       # Orchestration logic
├── model_wrappers.py # Model interfaces
├── config.py         # Configuration
├── logging_config.py # Structured logging
├── exceptions.py     # Error handling
├── middleware.py     # API middleware
└── metrics.py        # Prometheus metrics
```
