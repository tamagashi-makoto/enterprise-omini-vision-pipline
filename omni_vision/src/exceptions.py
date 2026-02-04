"""
Custom exceptions and error handling for enterprise-grade error responses.
"""
from typing import Any, Dict, Optional
from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes for the API."""

    # Client errors (4xx)
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_IMAGE = "INVALID_IMAGE"
    INVALID_MODE = "INVALID_MODE"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"

    # Server errors (5xx)
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_INFERENCE_ERROR = "MODEL_INFERENCE_ERROR"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class OmniVisionException(Exception):
    """Base exception for all Omni-Vision errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ClientError(OmniVisionException):
    """Base class for client errors (4xx)."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INVALID_REQUEST,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, status_code, details)


class ServerError(OmniVisionException):
    """Base class for server errors (5xx)."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, status_code, details)


# Specific client errors
class InvalidImageError(ClientError):
    """Raised when the uploaded image is invalid."""

    def __init__(self, message: str = "Invalid image file", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCode.INVALID_IMAGE, 400, details)


class InvalidModeError(ClientError):
    """Raised when an invalid mode is specified."""

    def __init__(self, mode: str, valid_modes: list):
        super().__init__(
            f"Invalid mode '{mode}'. Valid modes: {valid_modes}",
            ErrorCode.INVALID_MODE,
            400,
            {"provided_mode": mode, "valid_modes": valid_modes}
        )


class FileTooLargeError(ClientError):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, size: int, max_size: int):
        super().__init__(
            f"File size ({size} bytes) exceeds maximum allowed size ({max_size} bytes)",
            ErrorCode.FILE_TOO_LARGE,
            413,
            {"file_size": size, "max_size": max_size}
        )


class UnsupportedFormatError(ClientError):
    """Raised when file format is not supported."""

    def __init__(self, content_type: str, supported_formats: list):
        super().__init__(
            f"Unsupported content type '{content_type}'. Supported: {supported_formats}",
            ErrorCode.UNSUPPORTED_FORMAT,
            400,
            {"content_type": content_type, "supported_formats": supported_formats}
        )


# Specific server errors
class ModelLoadError(ServerError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            ErrorCode.MODEL_LOAD_ERROR,
            503,
            {"model": model_name, "reason": reason}
        )


class ModelInferenceError(ServerError):
    """Raised when model inference fails."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Model '{model_name}' inference failed: {reason}",
            ErrorCode.MODEL_INFERENCE_ERROR,
            500,
            {"model": model_name, "reason": reason}
        )


class PipelineError(ServerError):
    """Raised when pipeline execution fails."""

    def __init__(self, stage: str, reason: str):
        super().__init__(
            f"Pipeline error at stage '{stage}': {reason}",
            ErrorCode.PIPELINE_ERROR,
            500,
            {"stage": stage, "reason": reason}
        )


class ServiceUnavailableError(ServerError):
    """Raised when a required service is unavailable."""

    def __init__(self, service: str, reason: str = "Service temporarily unavailable"):
        super().__init__(
            f"{service}: {reason}",
            ErrorCode.SERVICE_UNAVAILABLE,
            503,
            {"service": service, "reason": reason}
        )
