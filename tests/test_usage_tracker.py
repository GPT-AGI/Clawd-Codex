"""Tests for usage_tracker module."""

import unittest

from src.usage_tracker import UsageTracker, UsageRecord, UsageStatistics


class TestUsageTracker(unittest.TestCase):
    """Test cases for usage tracking."""

    def test_create_tracker(self):
        """Test creating a usage tracker."""
        tracker = UsageTracker()
        self.assertEqual(len(tracker.records), 0)
        self.assertEqual(tracker.statistics.total_requests, 0)

    def test_create_tracker_with_budget(self):
        """Test creating a tracker with budget."""
        tracker = UsageTracker(budget_usd=10.0)
        self.assertEqual(tracker.budget_usd, 10.0)

    def test_record_usage(self):
        """Test recording token usage."""
        tracker = UsageTracker()

        record = tracker.record_usage(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
        )

        # Check record
        self.assertIsInstance(record, UsageRecord)
        self.assertEqual(record.provider, "anthropic")
        self.assertEqual(record.model, "claude-sonnet-4-6")
        self.assertEqual(record.input_tokens, 100)
        self.assertEqual(record.output_tokens, 50)
        self.assertGreater(record.cost_usd, 0)

        # Check statistics updated
        self.assertEqual(tracker.statistics.total_requests, 1)
        self.assertEqual(tracker.statistics.total_input_tokens, 100)
        self.assertEqual(tracker.statistics.total_output_tokens, 50)

    def test_multiple_records(self):
        """Test recording multiple usage events."""
        tracker = UsageTracker()

        tracker.record_usage("anthropic", "claude-sonnet-4-6", 100, 50)
        tracker.record_usage("openai", "gpt-4o", 200, 100)
        tracker.record_usage("anthropic", "claude-haiku-4-5", 150, 75)

        # Check totals
        self.assertEqual(tracker.statistics.total_requests, 3)
        self.assertEqual(tracker.statistics.total_input_tokens, 450)
        self.assertEqual(tracker.statistics.total_output_tokens, 225)

        # Check by-provider aggregation
        self.assertIn("anthropic", tracker.statistics.by_provider)
        self.assertIn("openai", tracker.statistics.by_provider)
        self.assertEqual(tracker.statistics.by_provider["anthropic"]["requests"], 2)
        self.assertEqual(tracker.statistics.by_provider["openai"]["requests"], 1)

    def test_cost_calculation(self):
        """Test cost calculation."""
        tracker = UsageTracker()

        # Test Anthropic pricing
        cost = tracker.calculate_cost("anthropic", "claude-sonnet-4-6", 1000, 500)
        # Input: (1000/1M) * $3 = $0.003
        # Output: (500/1M) * $15 = $0.0075
        # Total: $0.0105
        self.assertAlmostEqual(cost, 0.0105, places=4)

    def test_budget_exceeded(self):
        """Test budget exceeded detection."""
        tracker = UsageTracker(budget_usd=0.001)  # Very low budget

        # First request should succeed
        tracker.record_usage("anthropic", "claude-haiku-4-5", 10, 10)

        # Second request should fail
        with self.assertRaises(RuntimeError) as context:
            tracker.record_usage("anthropic", "claude-sonnet-4-6", 10000, 10000)

        self.assertIn("Budget exceeded", str(context.exception))

    def test_check_budget(self):
        """Test budget checking."""
        tracker = UsageTracker(budget_usd=1.0)

        # Initially not exceeded
        is_exceeded, percentage = tracker.check_budget()
        self.assertFalse(is_exceeded)
        self.assertEqual(percentage, 0.0)

        # Add some usage
        tracker.record_usage("anthropic", "claude-sonnet-4-6", 10000, 5000)

        # Check again
        is_exceeded, percentage = tracker.check_budget()
        self.assertFalse(is_exceeded)
        self.assertGreater(percentage, 0.0)

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        tracker = UsageTracker(budget_usd=10.0)
        tracker.record_usage("anthropic", "claude-sonnet-4-6", 100, 50)

        # Serialize
        data = tracker.to_dict()
        self.assertIn("records", data)
        self.assertIn("statistics", data)
        self.assertEqual(data["budget_usd"], 10.0)

        # Deserialize
        restored = UsageTracker.from_dict(data)
        self.assertEqual(restored.budget_usd, 10.0)
        self.assertEqual(len(restored.records), 1)
        self.assertEqual(restored.statistics.total_requests, 1)

    def test_statistics_markdown(self):
        """Test statistics markdown rendering."""
        tracker = UsageTracker()
        tracker.record_usage("anthropic", "claude-sonnet-4-6", 100, 50)

        md = tracker.statistics.as_markdown()
        self.assertIn("# Usage Statistics", md)
        self.assertIn("Total Requests: 1", md)
        self.assertIn("anthropic", md)


if __name__ == "__main__":
    unittest.main()
