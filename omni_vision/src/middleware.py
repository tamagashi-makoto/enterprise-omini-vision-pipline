"""
Enterprise middleware for API security, observability, and rate limiting.
"""
import time
import uuid
from typing import Callable, Optional
from collections import defaultdict
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from datetime import datetime, timedelta

from .logging_config import get_api_logger
from .config import config

logger = get_api_logger()


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    For production, consider using Redis-backed rate limiting.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: defaultdict = defaultdict(list)
        self._cleanup_interval = 60  # Clean up old entries every 60 seconds
        self._last_cleanup = time.time()

    def _cleanup_old_entries(self):
        """Remove entries older than 1 minute."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            cutoff = now - 60
            for key in list(self.requests.keys()):
                self.requests[key] = [
                    timestamp for timestamp in self.requests[key]
                    if timestamp > cutoff
                ]
                if not self.requests[key]:
                    del self.requests[key]
            self._last_cleanup = now

    def check_rate_limit(self, key: str) -> bool:
        """
        Check if the request is within rate limit.

        Args:
            key: Identifier for rate limiting (e.g., IP address)

        Returns:
            True if within limit, False otherwise
        """
        self._cleanup_old_entries()
        now = time.time()
        minute_ago = now - 60

        # Filter out requests older than 1 minute
        self.requests[key] = [
            timestamp for timestamp in self.requests[key]
            if timestamp > minute_ago
        ]

        if len(self.requests[key]) >= self.requests_per_minute:
            return False

        self.requests[key].append(now)
        return True

    def get_remaining_requests(self, key: str) -> int:
        """Get remaining requests for the given key."""
        self._cleanup_old_entries()
        return max(0, self.requests_per_minute - len(self.requests.get(key, [])))


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=config.api.rate_limit_per_minute
        )
    return _rate_limiter


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request context (ID, timestamp) to each request.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Add start time for latency tracking
        request.state.start_time = time.time()

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(
            time.time() - request.state.start_time
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    """

    def __init__(self, app: ASGIApp, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.rate_limiter = get_rate_limiter() if enabled else None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled or self.rate_limiter is None:
            return await call_next(request)

        # Use client IP for rate limiting
        client_ip = self._get_client_ip(request)

        if not self.rate_limiter.check_rate_limit(client_ip):
            remaining = self.rate_limiter.get_remaining_requests(client_ip)
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "extra_fields": {
                        "client_ip": client_ip,
                        "remaining": 0,
                        "path": request.url.path
                    }
                }
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": self.rate_limiter.requests_per_minute,
                    "remaining": remaining,
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.rate_limiter.requests_per_minute),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time()) + 60)
                }
            )

        response = await call_next(request)

        # Add rate limit info to headers
        remaining = self.rate_limiter.get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    """

    def __init__(self, app: ASGIApp, log_requests: bool = True, log_body: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_body = log_body

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.log_requests:
            return await call_next(request)

        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request
        logger.info(
            "Incoming request",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": self._get_client_ip(request),
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            }
        )

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time

            log_fields = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2)
            }

            if error:
                log_fields["error"] = error

            if status_code >= 400:
                logger.error("Request failed", extra={"extra_fields": log_fields})
            else:
                logger.info("Request completed", extra={"extra_fields": log_fields})

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]

        return response


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for limiting request content size.
    """

    def __init__(self, app: ASGIApp, max_size_mb: int = 50):
        super().__init__(app)
        self.max_size_bytes = max_size_mb * 1024 * 1024

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")

        if content_length:
            content_length = int(content_length)
            if content_length > self.max_size_bytes:
                logger.warning(
                    "Request too large",
                    extra={
                        "extra_fields": {
                            "content_length": content_length,
                            "max_size": self.max_size_bytes
                        }
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request body too large. Maximum size: {self.max_size_bytes // (1024 * 1024)}MB"
                )

        return await call_next(request)
