"""
Configuration management with environment variable support and validation.
"""
import os
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path


class ModelType(str, Enum):
    """Enumeration of supported computer vision models."""
    YOLO_V12 = "YOLOv12"
    RF_DETR = "RF-DETR"
    FLORENCE_2 = "Florence-2"
    SAM_3 = "SAM-3"


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class PipelineConfig:
    """Pipeline processing configuration."""

    # Thresholds
    density_threshold: int = field(default_factory=lambda: int(os.getenv("DENSITY_THRESHOLD", "15")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")))

    # BoxSet / Pipeline Config
    max_mask_boxes: int = field(default_factory=lambda: int(os.getenv("MAX_MASK_BOXES", "20")))
    nms_iou_threshold: float = field(default_factory=lambda: float(os.getenv("NMS_IOU_THRESHOLD", "0.5")))
    min_box_area_ratio: float = field(default_factory=lambda: float(os.getenv("MIN_BOX_AREA_RATIO", "0.0005")))

    # Mode Settings
    mode_default: str = field(default_factory=lambda: os.getenv("MODE_DEFAULT", "auto"))
    enable_florence_rerank: bool = field(default_factory=lambda: os.getenv("ENABLE_FLORENCE_RERANK", "false").lower() == "true")
    sam3_text_first: bool = field(default_factory=lambda: os.getenv("SAM3_TEXT_FIRST", "true").lower() == "true")
    fallback_if_no_mask: bool = field(default_factory=lambda: os.getenv("FALLBACK_IF_NO_MASK", "true").lower() == "true")

    def __post_init__(self):
        """Validate configuration values."""
        if self.density_threshold < 0:
            raise ValueError("DENSITY_THRESHOLD must be non-negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
        if self.max_mask_boxes < 1:
            raise ValueError("MAX_MASK_BOXES must be at least 1")
        if not 0 <= self.nms_iou_threshold <= 1:
            raise ValueError("NMS_IOU_THRESHOLD must be between 0 and 1")
        if self.mode_default not in ("auto", "query", "smart_query"):
            raise ValueError("MODE_DEFAULT must be 'auto', 'query', or 'smart_query'")


@dataclass
class ModelConfig:
    """Model-specific configuration."""

    # Latency estimates (seconds)
    latency_yolo: float = 0.05
    latency_rf_detr: float = 0.2
    latency_florence_2: float = 0.3
    latency_sam_3: float = 0.15

    # Model weights
    model_weights_dir: str = field(default_factory=lambda: os.getenv("MODEL_WEIGHTS_DIR", "src/weights/"))

    # Gemma3 Query Generator
    gemma3_enabled: bool = field(default_factory=lambda: os.getenv("GEMMA3_ENABLED", "true").lower() == "true")
    gemma3_model_id: str = field(default_factory=lambda: os.getenv("GEMMA3_MODEL_ID", "google/gemma-3-4b-it-qat-q4_0-gguf"))
    gemma3_model_file: str = field(default_factory=lambda: os.getenv("GEMMA3_MODEL_FILE", "gemma-3-4b-it-q4_0.gguf"))
    gemma3_max_queries: int = field(default_factory=lambda: int(os.getenv("GEMMA3_MAX_QUERIES", "5")))
    gemma3_context_size: int = field(default_factory=lambda: int(os.getenv("GEMMA3_CONTEXT_SIZE", "2048")))
    gemma3_gpu_layers: int = field(default_factory=lambda: int(os.getenv("GEMMA3_GPU_LAYERS", "-1")))


@dataclass
class APIConfig:
    """API configuration."""

    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))

    # Request limits
    max_request_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_REQUEST_SIZE_MB", "100")))
    max_image_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_SIZE_MB", "50")))
    request_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300")))

    # Rate limiting
    rate_limit_enabled: bool = field(default_factory=lambda: os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true")
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))

    # API versioning
    api_version: str = field(default_factory=lambda: os.getenv("API_VERSION", "v1"))

    def __post_init__(self):
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
        if self.max_image_size_mb < 1:
            raise ValueError("MAX_IMAGE_SIZE_MB must be at least 1")
        if self.request_timeout_seconds < 1:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be at least 1")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format_type: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))

    # Request logging
    log_requests: bool = field(default_factory=lambda: os.getenv("LOG_REQUESTS", "true").lower() == "true")
    log_request_body: bool = field(default_factory=lambda: os.getenv("LOG_REQUEST_BODY", "false").lower() == "true")


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration."""

    enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    path: str = field(default_factory=lambda: os.getenv("METRICS_PATH", "/metrics"))


@dataclass
class Config:
    """Central configuration for the Omni-Vision pipeline."""

    # Environment
    environment: Environment = field(default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "development")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Sub-configurations
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # HuggingFace
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    @classmethod
    def load_from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls()

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT


# Global configuration instance
config = Config.load_from_env()


# Legacy compatibility (keep old class-based access)
class ConfigLegacy:
    """Legacy Config class for backward compatibility."""

    DENSITY_THRESHOLD: int = 15
    CONFIDENCE_THRESHOLD: float = 0.5

    LATENCY_YOLO: float = 0.05
    LATENCY_RF_DETR: float = 0.2
    LATENCY_FLORENCE_2: float = 0.3
    LATENCY_SAM_3: float = 0.15

    MODEL_WEIGHTS_DIR: str = "src/weights/"

    MAX_MASK_BOXES: int = 20
    NMS_IOU_THRESHOLD: float = 0.5
    MIN_BOX_AREA_RATIO: float = 0.0005

    MODE_DEFAULT: str = "auto"
    ENABLE_FLORENCE_RERANK: bool = False
    SAM3_TEXT_FIRST: bool = True
    FALLBACK_IF_NO_MASK: bool = True

    GEMMA3_ENABLED: bool = True
    GEMMA3_MODEL_ID: str = "google/gemma-3-4b-it-qat-q4_0-gguf"
    GEMMA3_MODEL_FILE: str = "gemma-3-4b-it-q4_0.gguf"
    GEMMA3_MAX_QUERIES: int = 5
    GEMMA3_CONTEXT_SIZE: int = 2048
    GEMMA3_GPU_LAYERS: int = -1

    @classmethod
    def get(cls, key: str, default=None):
        """Get config value by key for backward compatibility."""
        return getattr(cls, key, default)


# Use legacy config for existing code compatibility
Config = ConfigLegacy
