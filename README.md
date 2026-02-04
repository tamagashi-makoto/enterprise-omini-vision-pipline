# Omni-Vision-Analytics

Enterprise-grade computer vision pipeline for object detection and segmentation with intelligent model orchestration.

## Features

- **Multi-model orchestration**: YOLOv12, RF-DETR, Florence-2, SAM3
- **Intelligent routing**: Selects models based on scene complexity
- **Text-guided segmentation**: Natural language object detection via Gemma3
- **Enterprise observability**: Structured JSON logging, Prometheus metrics
- **Security**: Rate limiting, CORS, request size limits
- **Health monitoring**: System resource tracking, model status

## Examples

### Background Removal
```bash
curl -X POST http://localhost:8000/analyze -F "file=@product.jpg"
```

### Privacy Masking
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@photo.jpg" \
  -F "text_query=face, license plate" \
  -F "mode=query"
```

### Natural Language Search
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@image.jpg" \
  -F "user_text=Find all vehicles in this image" \
  -F "mode=smart_query"
```

## Architecture

```
Input -> YOLOv12 -> RF-DETR (if dense) -> NMS -> Filter -> SAM3 -> Masks
         Fast        Dense                        Area
```

**Models**
- YOLOv12: Fast object detection
- RF-DETR: Dense scene detection
- Florence-2: Open-vocabulary detection
- SAM3: Pixel-level segmentation
- Gemma3: Natural language to query conversion

**Modes**
- `auto`: Detector-based flow → SAM3 (box prompts)
- `query`: Text-prompted → fallback to detectors
- `smart_query`: Natural language → Gemma3 → SAM3

## Installation

Docker (recommended):
```bash
docker-compose up --build
```

Local:
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Requirements: Python 3.10+, CUDA GPU (recommended), HF_TOKEN for SAM3.

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | deployment environment |
| `LOG_LEVEL` | INFO | logging level |
| `LOG_FORMAT` | json | json or text |
| `RATE_LIMIT_ENABLED` | true | enable rate limiting |
| `RATE_LIMIT_PER_MINUTE` | 60 | requests per minute |
| `MAX_IMAGE_SIZE_MB` | 50 | max upload size |
| `METRICS_ENABLED` | true | enable Prometheus metrics |

See `src/config.py` for all options.

## API

### POST /analyze

Analyze an image.

| Param | Type | Description |
|-------|------|-------------|
| file | file | Image to analyze (required) |
| text_query | string | Direct query for query mode |
| user_text | string | Natural language for smart_query mode |
| mode | string | auto, query, or smart_query |
| mask_format | string | rle, png_base64, or none |

### GET /health

Health check with system metrics and model status.

### GET /metrics

Prometheus metrics endpoint.

## Monitoring

**Structured Logging** (JSON format):
```json
{
  "timestamp": "2024-01-01T00:00:00",
  "level": "INFO",
  "logger": "omni_vision.pipeline",
  "message": "Pipeline completed",
  "request_id": "abc-123",
  "objects_detected": 5,
  "duration_seconds": 1.23
}
```

**Prometheus Metrics**:
- `http_requests_total`: Request count by status
- `http_request_duration_seconds`: Request latency
- `pipeline_duration_seconds`: Pipeline execution time
- `model_inference_duration_seconds`: Model latency
- `objects_detected_total`: Total objects detected
- `gpu_memory_bytes`: GPU memory usage

## Deployment

### Docker Compose
```yaml
services:
  api:
    image: omni-vision:latest
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=WARNING
      - RATE_LIMIT_PER_MINUTE=120
    ports:
      - "8000:8000"
```

### Kubernetes
```yaml
env:
  - name: ENVIRONMENT
    value: "production"
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: huggingface
        key: token
resources:
  limits:
    nvidia.com/gpu: 1
```

## Testing

```bash
# Unit tests
pytest tests/test_pipeline_unit.py -v

# API tests
pytest tests/test_api.py -v

# Integration (requires GPU)
RUN_INTEGRATION_TESTS=1 pytest -m integration
```
