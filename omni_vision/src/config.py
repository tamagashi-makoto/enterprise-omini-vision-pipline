from enum import Enum

class ModelType(str, Enum):
    """Enumeration of supported computer vision models."""
    YOLO_V12 = "YOLOv12"
    RF_DETR = "RF-DETR"
    DINO_X = "DINO-X"
    SAM_3 = "SAM-3"

class Config:
    """Central configuration for the Omni-Vision pipeline."""
    
    # Thresholds
    DENSITY_THRESHOLD: int = 15
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Paths
    WEIGHTS_DIR: str = "weights"
    PATH_YOLO: str = "weights/yolov12s.pt"          # YOLOv12 (User specified)
    PATH_RF_DETR: str = "weights/rf_detr.pt"       # RF-DETR (Model ID)
    PATH_FLORENCE_2: str = "microsoft/Florence-2-base" # Florence-2 (HuggingFace ID)
    PATH_SAM_3: str = "weights/sam3.pt"           # SAM3 (Requires CUDA, HF auth for download)
    
    # API Keys
    ROBOFLOW_API_KEY: str = "uUgID4rbpfXeUdbNClkg" # From user screenshot
    
    # Latency values are now dependent on actual inference, but we keep init latency for safety
    LATENCY_YOLO: float = 0.0
    LATENCY_RF_DETR: float = 0.0
    LATENCY_DINO_X: float = 0.0
    LATENCY_SAM_3: float = 0.0
