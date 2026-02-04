"""
Prometheus metrics collection for monitoring and observability.
"""
import time
import threading
from typing import Dict, List, Optional, Callable
from functools import wraps
from collections import defaultdict

# Try to import prometheus_client, make it optional
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsRegistry:
    """
    Thread-safe metrics registry for Prometheus metrics.
    Falls back to in-memory storage if prometheus_client is not available.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._labels: Dict[str, Dict[str, str]] = {}

        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._prometheus_counters: Dict[str, Counter] = {}
            self._prometheus_gauges: Dict[str, Gauge] = {}
            self._prometheus_histograms: Dict[str, Histogram] = {}
        else:
            self.registry = None

    def counter(
        self,
        name: str,
        documentation: str,
        labels: Optional[List[str]] = None
    ) -> "CounterMetric":
        """Create or get a counter metric."""
        with self._lock:
            if PROMETHEUS_AVAILABLE and name not in self._prometheus_counters:
                self._prometheus_counters[name] = Counter(
                    name,
                    documentation,
                    list(labels) if labels else [],
                    registry=self.registry
                )
            return CounterMetric(self, name, labels or [])

    def gauge(
        self,
        name: str,
        documentation: str,
        labels: Optional[List[str]] = None
    ) -> "GaugeMetric":
        """Create or get a gauge metric."""
        with self._lock:
            if PROMETHEUS_AVAILABLE and name not in self._prometheus_gauges:
                self._prometheus_gauges[name] = Gauge(
                    name,
                    documentation,
                    list(labels) if labels else [],
                    registry=self.registry
                )
            return GaugeMetric(self, name, labels or [])

    def histogram(
        self,
        name: str,
        documentation: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[List[str]] = None
    ) -> "HistogramMetric":
        """Create or get a histogram metric."""
        with self._lock:
            if PROMETHEUS_AVAILABLE and name not in self._prometheus_histograms:
                self._prometheus_histograms[name] = Histogram(
                    name,
                    documentation,
                    buckets=buckets,
                    list(labels) if labels else [],
                    registry=self.registry
                )
            return HistogramMetric(self, name, labels or [])

    def inc_counter(self, name: str, value: float = 1.0, label_values: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        with self._lock:
            key = self._make_key(name, label_values)
            self._counters[key] += value

            if PROMETHEUS_AVAILABLE and name in self._prometheus_counters:
                if label_values:
                    self._prometheus_counters[name].labels(**label_values).inc(value)
                else:
                    self._prometheus_counters[name].inc(value)

    def set_gauge(self, name: str, value: float, label_values: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        with self._lock:
            key = self._make_key(name, label_values)
            self._gauges[key] = value

            if PROMETHEUS_AVAILABLE and name in self._prometheus_gauges:
                if label_values:
                    self._prometheus_gauges[name].labels(**label_values).set(value)
                else:
                    self._prometheus_gauges[name].set(value)

    def observe_histogram(self, name: str, value: float, label_values: Optional[Dict[str, str]] = None):
        """Observe a value for a histogram."""
        with self._lock:
            key = self._make_key(name, label_values)
            self._histograms[key].append(value)

            if PROMETHEUS_AVAILABLE and name in self._prometheus_histograms:
                if label_values:
                    self._prometheus_histograms[name].labels(**label_values).observe(value)
                else:
                    self._prometheus_histograms[name].observe(value)

    def _make_key(self, name: str, label_values: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric with labels."""
        if label_values:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(label_values.items()))
            return f"{name}{{{label_str}}}"
        return name

    def generate_text(self) -> str:
        """Generate Prometheus text format metrics."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode("utf-8")

        # Fallback: simple text format
        lines = []
        for key, value in self._counters.items():
            lines.append(f"# TYPE {key} counter")
            lines.append(f"{key} {value}")
        for key, value in self._gauges.items():
            lines.append(f"# TYPE {key} gauge")
            lines.append(f"{key} {value}")
        for key, values in self._histograms.items():
            lines.append(f"# TYPE {key} histogram")
            lines.append(f"{key}_count {len(values)}")
            if values:
                lines.append(f"{key}_sum {sum(values)}")
        return "\n".join(lines)


class CounterMetric:
    """Counter metric wrapper."""

    def __init__(self, registry: MetricsRegistry, name: str, labels: List[str]):
        self.registry = registry
        self.name = name
        self.labels = labels

    def inc(self, value: float = 1.0, **label_values):
        """Increment the counter."""
        if self.labels and not label_values:
            raise ValueError(f"Label values required for {self.name}")
        self.registry.inc_counter(self.name, value, label_values if label_values else None)


class GaugeMetric:
    """Gauge metric wrapper."""

    def __init__(self, registry: MetricsRegistry, name: str, labels: List[str]):
        self.registry = registry
        self.name = name
        self.labels = labels

    def set(self, value: float, **label_values):
        """Set the gauge value."""
        if self.labels and not label_values:
            raise ValueError(f"Label values required for {self.name}")
        self.registry.set_gauge(self.name, value, label_values if label_values else None)

    def inc(self, value: float = 1.0, **label_values):
        """Increment the gauge."""
        if self.labels and not label_values:
            raise ValueError(f"Label values required for {self.name}")
        current = self.registry._gauges.get(self.name, 0)
        self.registry.set_gauge(self.name, current + value, label_values if label_values else None)

    def dec(self, value: float = 1.0, **label_values):
        """Decrement the gauge."""
        self.inc(-value, **label_values)


class HistogramMetric:
    """Histogram metric wrapper."""

    def __init__(self, registry: MetricsRegistry, name: str, labels: List[str]):
        self.registry = registry
        self.name = name
        self.labels = labels

    def observe(self, value: float, **label_values):
        """Observe a value."""
        if self.labels and not label_values:
            raise ValueError(f"Label values required for {self.name}")
        self.registry.observe_histogram(self.name, value, label_values if label_values else None)


# Global metrics registry
registry = MetricsRegistry()


# Define standard metrics
http_requests_total = registry.counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = registry.histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
    labels=["method", "endpoint"]
)

http_requests_active = registry.gauge(
    "http_requests_active",
    "Active HTTP requests"
)

pipeline_requests_total = registry.counter(
    "pipeline_requests_total",
    "Total pipeline requests",
    ["mode", "status"]
)

pipeline_duration_seconds = registry.histogram(
    "pipeline_duration_seconds",
    "Pipeline execution time",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    labels=["mode"]
)

model_inference_duration_seconds = registry.histogram(
    "model_inference_duration_seconds",
    "Model inference time",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    labels=["model"]
)

model_load_errors_total = registry.counter(
    "model_load_errors_total",
    "Total model load errors",
    ["model"]
)

objects_detected_total = registry.counter(
    "objects_detected_total",
    "Total objects detected"
)

masks_generated_total = registry.counter(
    "masks_generated_total",
    "Total masks generated"
)

gpu_memory_bytes = registry.gauge(
    "gpu_memory_bytes",
    "GPU memory usage",
    ["device"]
)


def track_time(metric: HistogramMetric, **label_values):
    """
    Decorator to track function execution time.

    Usage:
        @track_time(pipeline_duration_seconds, mode="auto")
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                metric.observe(time.time() - start_time, **label_values)
                return result
            except Exception:
                metric.observe(time.time() - start_time, **label_values)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                metric.observe(time.time() - start_time, **label_values)
                return result
            except Exception:
                metric.observe(time.time() - start_time, **label_values)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def update_gpu_metrics():
    """Update GPU-related metrics."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                gpu_memory_bytes.set(memory_allocated, device=f"cuda:{i}", type="allocated")
                gpu_memory_bytes.set(memory_reserved, device=f"cuda:{i}", type="reserved")
    except ImportError:
        pass
