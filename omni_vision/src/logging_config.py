"""
Structured logging configuration for enterprise observability.
"""
import logging
import sys
import json
from typing import Any, Dict
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ContextualLogger(logging.LoggerAdapter):
    """Logger adapter for adding contextual fields."""

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        extra = kwargs.get("extra", {})
        if hasattr(self, "extra"):
            extra = {**extra, **self.extra}
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: str = None
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: "json" or "text"
        log_file: Optional file path for logging output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if format_type == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )

    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)

    for handler in handlers:
        root_logger.addHandler(handler)


def get_logger(name: str, **context: Any) -> ContextualLogger:
    """
    Get a logger with optional context fields.

    Args:
        name: Logger name
        **context: Additional fields to include in all log messages

    Returns:
        ContextualLogger instance
    """
    logger = logging.getLogger(name)
    return ContextualLogger(logger, extra={"extra_fields": context})


# Pipeline-specific loggers
def get_pipeline_logger(request_id: str = None) -> ContextualLogger:
    """Get logger for pipeline operations."""
    context = {"component": "pipeline"}
    if request_id:
        context["request_id"] = request_id
    return get_logger("omni_vision.pipeline", **context)


def get_model_logger(model_name: str) -> ContextualLogger:
    """Get logger for model operations."""
    return get_logger("omni_vision.models", model=model_name)


def get_api_logger() -> ContextualLogger:
    """Get logger for API operations."""
    return get_logger("omni_vision.api", component="api")
