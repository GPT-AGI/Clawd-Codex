"""
Token usage and cost tracking module.

Provides comprehensive tracking of:
- Input/output token counts per request
- Cumulative usage statistics per session
- Cost estimation based on provider pricing
- Usage alerts and budget management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers.base import Provider

# Pricing per 1M tokens (USD)
# Updated as of 2025
PRICING = {
    "anthropic": {
        "claude-opus-4-6": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    },
    "openai": {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    "zhipuai": {
        "glm-4-plus": {"input": 0.05, "output": 0.05},  # RMB per 1K tokens
        "glm-4-flash": {"input": 0.001, "output": 0.001},
    },
}


@dataclass(frozen=True)
class UsageRecord:
    """Single usage record for one API call."""

    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class UsageStatistics:
    """Aggregate usage statistics."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    by_provider: dict[str, dict[str, int | float]] = field(default_factory=dict)
    by_model: dict[str, dict[str, int | float]] = field(default_factory=dict)

    def add_record(self, record: UsageRecord) -> None:
        """Add a usage record to statistics."""
        self.total_requests += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_cost_usd += record.cost_usd

        # Aggregate by provider
        if record.provider not in self.by_provider:
            self.by_provider[record.provider] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        self.by_provider[record.provider]["requests"] += 1
        self.by_provider[record.provider]["input_tokens"] += record.input_tokens
        self.by_provider[record.provider]["output_tokens"] += record.output_tokens
        self.by_provider[record.provider]["cost_usd"] += record.cost_usd

        # Aggregate by model
        model_key = f"{record.provider}/{record.model}"
        if model_key not in self.by_model:
            self.by_model[model_key] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        self.by_model[model_key]["requests"] += 1
        self.by_model[model_key]["input_tokens"] += record.input_tokens
        self.by_model[model_key]["output_tokens"] += record.output_tokens
        self.by_model[model_key]["cost_usd"] += record.cost_usd

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_provider": self.by_provider,
            "by_model": self.by_model,
        }

    def as_markdown(self) -> str:
        """Render as markdown report."""
        lines = [
            "# Usage Statistics",
            "",
            "## Summary",
            f"- Total Requests: {self.total_requests}",
            f"- Total Input Tokens: {self.total_input_tokens:,}",
            f"- Total Output Tokens: {self.total_output_tokens:,}",
            f"- Total Cost: ${self.total_cost_usd:.4f}",
            "",
            "## By Provider",
        ]

        for provider, stats in sorted(self.by_provider.items()):
            lines.append(f"### {provider}")
            lines.append(f"- Requests: {stats['requests']}")
            lines.append(f"- Input Tokens: {stats['input_tokens']:,}")
            lines.append(f"- Output Tokens: {stats['output_tokens']:,}")
            lines.append(f"- Cost: ${stats['cost_usd']:.4f}")
            lines.append("")

        lines.append("## By Model")
        for model, stats in sorted(self.by_model.items()):
            lines.append(f"### {model}")
            lines.append(f"- Requests: {stats['requests']}")
            lines.append(f"- Input Tokens: {stats['input_tokens']:,}")
            lines.append(f"- Output Tokens: {stats['output_tokens']:,}")
            lines.append(f"- Cost: ${stats['cost_usd']:.4f}")
            lines.append("")

        return "\n".join(lines)


class UsageTracker:
    """Tracks token usage and costs."""

    def __init__(self, budget_usd: float | None = None):
        """
        Initialize usage tracker.

        Args:
            budget_usd: Optional budget limit in USD
        """
        self.records: list[UsageRecord] = []
        self.statistics = UsageStatistics()
        self.budget_usd = budget_usd

    def calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            provider: Provider name (e.g., "anthropic")
            model: Model name (e.g., "claude-sonnet-4-6")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Get pricing
        provider_pricing = PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        timestamp: str | None = None,
    ) -> UsageRecord:
        """
        Record token usage.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            timestamp: Optional timestamp (defaults to now)

        Returns:
            UsageRecord

        Raises:
            RuntimeError: If budget exceeded
        """
        # Calculate cost
        cost = self.calculate_cost(provider, model, input_tokens, output_tokens)

        # Create record
        record = UsageRecord(
            timestamp=timestamp or datetime.now().isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        # Check budget
        if self.budget_usd is not None:
            projected_cost = self.statistics.total_cost_usd + cost
            if projected_cost > self.budget_usd:
                raise RuntimeError(
                    f"Budget exceeded: ${projected_cost:.4f} > ${self.budget_usd:.2f}"
                )

        # Add to records and statistics
        self.records.append(record)
        self.statistics.add_record(record)

        return record

    def get_statistics(self) -> UsageStatistics:
        """Get usage statistics."""
        return self.statistics

    def check_budget(self) -> tuple[bool, float]:
        """
        Check if budget is exceeded.

        Returns:
            Tuple of (is_exceeded, percentage_used)
        """
        if self.budget_usd is None:
            return False, 0.0

        percentage = (self.statistics.total_cost_usd / self.budget_usd) * 100
        return self.statistics.total_cost_usd > self.budget_usd, percentage

    def to_dict(self) -> dict:
        """Convert tracker state to dictionary."""
        return {
            "records": [r.to_dict() for r in self.records],
            "statistics": self.statistics.to_dict(),
            "budget_usd": self.budget_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UsageTracker:
        """Create tracker from dictionary."""
        tracker = cls(budget_usd=data.get("budget_usd"))

        for record_data in data.get("records", []):
            record = UsageRecord(**record_data)
            tracker.records.append(record)
            tracker.statistics.add_record(record)

        return tracker
