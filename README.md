# Omni-Vision-Analytics

Vision pipeline for object detection and segmentation. Routes to different models depending on scene density. SAM3 generates final masks.

## Examples

### Background Removal
```bash
curl -X POST http://localhost:8000/analyze -F "file=@product.jpg"
```

### Privacy Masking
```bash
curl -X POST http://localhost:8000/analyze -F "file=@photo.jpg" -F "text_query=face, license plate" -F "mode=query"
```

### Defect Detection
```bash
curl -X POST http://localhost:8000/analyze -F "file=@pcb.jpg" -F "text_query=scratch, crack" -F "mode=query"
```

## Architecture

```
Input -> YOLOv12 -> RF-DETR (if dense) -> NMS -> Filter -> SAM3 -> Masks
         Fast        Dense                        Area
```

**Models**
- YOLOv12: Fast object detection
- RF-DETR: Dense scenes (triggered when YOLO detects > DENSITY_THRESHOLD objects)
- Florence-2: Text-guided reranking (optional)
- SAM3: Segmentation masks

**Modes**
- `auto`: YOLO first, then RF-DETR if needed
- `query`: SAM3 text prompt first, falls back to detectors

## Install

Docker:
```bash
docker-compose up --build
```

Local:
```bash
pip install -r requirements.txt
HF_TOKEN=your_token python -m uvicorn src.main:app --host 0 --port 8000
```

Requires Python 3.10+, CUDA GPU recommended, HF_TOKEN for SAM3 access.

## API

### POST /analyze

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | Yes | Image to analyze |
| text_query | string | No | Search query for query mode |
| mode | string | No | auto or query (default: auto) |

Returns:
```json
{
  "meta": {"processing_mode": "YOLO(5)", "objects_detected": 5},
  "detections": [{"label": "person", "confidence": 0.95, "box": [10,20,100,200], "has_mask": true}],
  "masks_generated": 5
}
```

### GET /health

System health check.

## Testing

```bash
# unit tests (mocked, no gpu)
pytest tests/test_pipeline_unit.py -v

# api tests
pytest tests/test_api.py -v

# integration (needs gpu)
RUN_INTEGRATION_TESTS=1 pytest -m integration
```

## Config

Edit `src/config.py`:
- `MAX_MASK_BOXES`: max boxes for SAM3 (default 20)
- `NMS_IOU_THRESHOLD`: IoU threshold for merging boxes (0.5)
- `DENSITY_THRESHOLD`: triggers RF-DETR when YOLO finds more than this (15)
- `SAM3_TEXT_FIRST`: try text prompt before detectors in query mode (True)
- `ENABLE_FLORENCE_RERANK`: enable Florence-2 reranking (False)
