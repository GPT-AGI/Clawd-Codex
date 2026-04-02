"""Tests for observability module."""

import unittest
from pathlib import Path
from unittest.mock import Mock

from src.observability import (
    PerformanceMetric,
    RequestLog,
    ErrorLog,
    PerformanceMonitor,
    RequestLogger,
    ErrorTracker,
    ObservabilitySystem,
    create_observability_system,
)


class TestObservability(unittest.TestCase):
    """Test cases for observability system."""

    def test_create_performance_metric(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(name="test", value=100.0)

        self.assertEqual(metric.name, "test")
        self.assertEqual(metric.value, 100.0)
        self.assertEqual(metric.unit, "ms")

    def test_create_request_log(self):
        """Test creating a request log."""
        log = RequestLog(
            request_id="req-123",
            method="chat",
            provider="anthropic",
            model="claude-sonnet-4-6",
        )

        self.assertEqual(log.request_id, "req-123")
        self.assertEqual(log.provider, "anthropic")

    def test_create_error_log(self):
        """Test creating an error log."""
        error = ValueError("test error")
        log = ErrorLog(
            error_type="ValueError",
            error_message="test error",
        )

        self.assertEqual(log.error_type, "ValueError")
        self.assertEqual(log.error_message, "test error")

    def test_create_performance_monitor(self):
        """Test creating a performance monitor."""
        monitor = PerformanceMonitor()

        self.assertEqual(len(monitor.metrics), 0)

    def test_record_metric(self):
        """Test recording a metric."""
        monitor = PerformanceMonitor()
        monitor.record("test_metric", 50.0)

        self.assertEqual(len(monitor.metrics), 1)
        self.assertEqual(monitor.metrics[0].name, "test_metric")

    def test_timer_context(self):
        """Test timer context manager."""
        monitor = PerformanceMonitor()

        with monitor.timer("test_timer"):
            pass  # Do nothing

        metrics = monitor.get_metrics("test_timer")
        self.assertEqual(len(metrics), 1)
        self.assertGreater(metrics[0].value, 0)

    def test_get_metrics_filtered(self):
        """Test getting filtered metrics."""
        monitor = PerformanceMonitor()
        monitor.record("metric1", 10.0)
        monitor.record("metric2", 20.0)
        monitor.record("metric1", 30.0)

        metrics = monitor.get_metrics("metric1")

        self.assertEqual(len(metrics), 2)

    def test_get_average(self):
        """Test getting average metric value."""
        monitor = PerformanceMonitor()
        monitor.record("test", 10.0)
        monitor.record("test", 20.0)
        monitor.record("test", 30.0)

        avg = monitor.get_average("test")

        self.assertEqual(avg, 20.0)

    def test_create_request_logger(self):
        """Test creating a request logger."""
        logger = RequestLogger()

        self.assertEqual(len(logger.logs), 0)

    def test_log_request(self):
        """Test logging a request."""
        logger = RequestLogger()
        logger.log_request(
            request_id="req-1",
            method="chat",
            provider="test",
            model="test-model",
        )

        self.assertEqual(len(logger.logs), 1)

    def test_get_logs_filtered(self):
        """Test getting filtered logs."""
        logger = RequestLogger()
        logger.log_request("req-1", "chat", "anthropic", "claude")
        logger.log_request("req-2", "chat", "openai", "gpt-4")

        logs = logger.get_logs(provider="anthropic")

        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].provider, "anthropic")

    def test_create_error_tracker(self):
        """Test creating an error tracker."""
        tracker = ErrorTracker()

        self.assertEqual(len(tracker.errors), 0)

    def test_track_error(self):
        """Test tracking an error."""
        tracker = ErrorTracker()
        error = ValueError("test error")

        tracker.track_error(error, context={"test": "context"})

        self.assertEqual(len(tracker.errors), 1)
        self.assertEqual(tracker.errors[0].error_type, "ValueError")

    def test_get_errors_filtered(self):
        """Test getting filtered errors."""
        tracker = ErrorTracker()
        tracker.track_error(ValueError("error1"))
        tracker.track_error(RuntimeError("error2"))
        tracker.track_error(ValueError("error3"))

        errors = tracker.get_errors(error_type="ValueError")

        self.assertEqual(len(errors), 2)

    def test_create_observability_system(self):
        """Test creating observability system."""
        system = create_observability_system()

        self.assertIsInstance(system, ObservabilitySystem)
        self.assertIsInstance(system.performance, PerformanceMonitor)
        self.assertIsInstance(system.requests, RequestLogger)
        self.assertIsInstance(system.errors, ErrorTracker)

    def test_track_request_integration(self):
        """Test tracking a complete request."""
        system = ObservabilitySystem()

        system.track_request(
            request_id="req-1",
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            duration_ms=500.0,
        )

        logs = system.requests.get_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].input_tokens, 100)

    def test_get_summary(self):
        """Test getting observability summary."""
        system = ObservabilitySystem()

        system.track_request("req-1", "anthropic", "claude", 100, 50, 100.0)
        system.track_request("req-2", "openai", "gpt-4", 200, 100, 200.0, status="error")

        summary = system.get_summary()

        self.assertEqual(summary["total_requests"], 2)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["total_input_tokens"], 300)
        self.assertEqual(summary["total_output_tokens"], 150)


if __name__ == "__main__":
    unittest.main()
