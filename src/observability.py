"""
Observability and debugging tools for Clawd Codex.

Provides functionality to:
- Performance monitoring
- Request/response logging
- Token usage tracking
- Error tracking
- Debug mode support
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PerformanceMetric:
    """Performance metric record."""

    name: str
    value: float
    unit: str = "ms"
    timestamp: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RequestLog:
    """Request log entry."""

    request_id: str
    method: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    status: str = "success"
    timestamp: str = ""
    error: str | None = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ErrorLog:
    """Error log entry."""

    error_type: str
    error_message: str
    stack_trace: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PerformanceMonitor:
    """Monitors performance metrics."""

    def __init__(self, max_metrics: int = 1000):
        """Initialize monitor."""
        self.max_metrics = max_metrics
        self.metrics: list[PerformanceMetric] = []

    def record(self, name: str, value: float, unit: str = "ms", **tags) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(name=name, value=value, unit=unit, tags=tags)

        self.metrics.append(metric)

        # Trim old metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics :]

    def timer(self, name: str, **tags):
        """Context manager for timing operations."""
        return _TimerContext(self, name, tags)

    def get_metrics(self, name: str | None = None) -> list[PerformanceMetric]:
        """Get metrics, optionally filtered by name."""
        if name:
            return [m for m in self.metrics if m.name == name]
        return self.metrics

    def get_average(self, name: str) -> float | None:
        """Get average value for a metric."""
        metrics = self.get_metrics(name)
        if not metrics:
            return None
        return sum(m.value for m in metrics) / len(metrics)


class _TimerContext:
    """Timer context manager."""

    def __init__(self, monitor: PerformanceMonitor, name: str, tags: dict):
        self.monitor = monitor
        self.name = name
        self.tags = tags
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        duration = (time.time() - self.start_time) * 1000
        self.monitor.record(self.name, duration, "ms", **self.tags)


class RequestLogger:
    """Logs API requests."""

    def __init__(self, log_file: Path | None = None, max_logs: int = 1000):
        """Initialize logger."""
        self.log_file = log_file
        self.max_logs = max_logs
        self.logs: list[RequestLog] = []

    def log_request(
        self,
        request_id: str,
        method: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0.0,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        """Log a request."""
        log = RequestLog(
            request_id=request_id,
            method=method,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            status=status,
            error=error,
        )

        self.logs.append(log)

        # Trim old logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs :]

        # Write to file if configured
        if self.log_file:
            self._write_log(log)

    def _write_log(self, log: RequestLog) -> None:
        """Write log to file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.log_file, "a") as f:
                f.write(json.dumps(log.__dict__) + "\n")
        except OSError:
            pass  # Ignore file errors

    def get_logs(self, provider: str | None = None) -> list[RequestLog]:
        """Get logs, optionally filtered by provider."""
        if provider:
            return [log for log in self.logs if log.provider == provider]
        return self.logs


class ErrorTracker:
    """Tracks errors."""

    def __init__(self, max_errors: int = 100):
        """Initialize tracker."""
        self.max_errors = max_errors
        self.errors: list[ErrorLog] = []

    def track_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Track an error."""
        import traceback

        error_log = ErrorLog(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
        )

        self.errors.append(error_log)

        # Trim old errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors :]

    def get_errors(self, error_type: str | None = None) -> list[ErrorLog]:
        """Get errors, optionally filtered by type."""
        if error_type:
            return [e for e in self.errors if e.error_type == error_type]
        return self.errors


class ObservabilitySystem:
    """Complete observability system."""

    def __init__(self, log_dir: Path | None = None):
        """
        Initialize observability system.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir or Path(".clawd/logs")

        self.performance = PerformanceMonitor()
        self.requests = RequestLogger(
            log_file=self.log_dir / "requests.log" if log_dir else None
        )
        self.errors = ErrorTracker()

    def track_request(
        self,
        request_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        """Track a complete request."""
        self.requests.log_request(
            request_id=request_id,
            method="chat",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            status=status,
            error=error,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get observability summary."""
        request_logs = self.requests.get_logs()

        return {
            "total_requests": len(request_logs),
            "successful_requests": sum(1 for r in request_logs if r.status == "success"),
            "failed_requests": sum(1 for r in request_logs if r.status == "error"),
            "total_errors": len(self.errors.errors),
            "average_request_duration": (
                sum(r.duration_ms for r in request_logs) / len(request_logs)
                if request_logs
                else 0
            ),
            "total_input_tokens": sum(r.input_tokens for r in request_logs),
            "total_output_tokens": sum(r.output_tokens for r in request_logs),
        }


def create_observability_system(log_dir: Path | None = None) -> ObservabilitySystem:
    """Create an observability system."""
    return ObservabilitySystem(log_dir=log_dir)
