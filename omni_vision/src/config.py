"""
Configuration for the Omni-Vision pipeline.
"""
from enum import Enum


class ModelType(str, Enum):
    """Enumeration of supported computer vision models."""
    YOLO_V12 = "YOLOv12"
    RF_DETR = "RF-DETR"
    FLORENCE_2 = "Florence-2"  # Replaced DINO-X
    SAM_3 = "SAM-3"


class Config:
    """Central configuration for the Omni-Vision pipeline."""
    
    # Thresholds
    DENSITY_THRESHOLD: int = 15
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Model Latencies (estimated in seconds)
    LATENCY_YOLO: float = 0.05
    LATENCY_RF_DETR: float = 0.2
    LATENCY_FLORENCE_2: float = 0.3  # Replaced DINO-X
    LATENCY_SAM_3: float = 0.15

    # Paths
    MODEL_WEIGHTS_DIR: str = "src/weights/"
