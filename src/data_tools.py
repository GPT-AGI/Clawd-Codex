"""
Data engineering and ETL tools for Clawd Codex.

Provides functionality to:
- Read/write various data formats (CSV, JSON, Parquet)
- Transform and clean data
- Execute data pipelines
- Generate data profiles
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DataProfile:
    """Data profile summary."""

    row_count: int
    column_count: int
    columns: dict[str, dict[str, Any]]
    missing_values: dict[str, int]
    sample_rows: list[dict[str, Any]]


class DataReader:
    """Reads data from various formats."""

    @staticmethod
    def read_csv(
        path: Path | str,
        delimiter: str = ",",
        has_header: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Read CSV file.

        Args:
            path: File path
            delimiter: Column delimiter
            has_header: Whether file has header

        Returns:
            List of row dictionaries
        """
        path = Path(path)
        rows = []

        with open(path, newline="", encoding="utf-8") as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = list(reader)
            else:
                reader = csv.reader(f, delimiter=delimiter)
                for row in reader:
                    rows.append({str(i): val for i, val in enumerate(row)})

        return rows

    @staticmethod
    def read_json(path: Path | str) -> Any:
        """
        Read JSON file.

        Args:
            path: File path

        Returns:
            Parsed JSON data
        """
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
        """
        Read JSON Lines file.

        Args:
            path: File path

        Returns:
            List of JSON objects
        """
        path = Path(path)
        rows = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        return rows


class DataWriter:
    """Writes data to various formats."""

    @staticmethod
    def write_csv(
        data: list[dict[str, Any]],
        path: Path | str,
        delimiter: str = ",",
    ) -> None:
        """
        Write CSV file.

        Args:
            data: List of row dictionaries
            path: File path
            delimiter: Column delimiter
        """
        if not data:
            return

        path = Path(path)
        fieldnames = list(data[0].keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def write_json(
        data: Any,
        path: Path | str,
        indent: int = 2,
    ) -> None:
        """
        Write JSON file.

        Args:
            data: Data to write
            path: File path
            indent: JSON indentation
        """
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def write_jsonl(
        data: list[dict[str, Any]],
        path: Path | str,
    ) -> None:
        """
        Write JSON Lines file.

        Args:
            data: List of JSON objects
            path: File path
        """
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class DataTransformer:
    """Transforms data."""

    @staticmethod
    def map_columns(
        data: list[dict[str, Any]],
        mapping: dict[str, str],
    ) -> list[dict[str, Any]]:
        """
        Rename columns.

        Args:
            data: Input data
            mapping: Column mapping {old: new}

        Returns:
            Transformed data
        """
        result = []

        for row in data:
            new_row = {}
            for key, value in row.items():
                new_key = mapping.get(key, key)
                new_row[new_key] = value
            result.append(new_row)

        return result

    @staticmethod
    def filter_rows(
        data: list[dict[str, Any]],
        predicate: Callable[[dict[str, Any]], bool],
    ) -> list[dict[str, Any]]:
        """
        Filter rows by predicate.

        Args:
            data: Input data
            predicate: Filter function

        Returns:
            Filtered data
        """
        return [row for row in data if predicate(row)]

    @staticmethod
    def transform_values(
        data: list[dict[str, Any]],
        column: str,
        transformer: Callable[[Any], Any],
    ) -> list[dict[str, Any]]:
        """
        Transform values in a column.

        Args:
            data: Input data
            column: Column name
            transformer: Transformation function

        Returns:
            Transformed data
        """
        result = []

        for row in data:
            new_row = row.copy()
            if column in new_row:
                new_row[column] = transformer(new_row[column])
            result.append(new_row)

        return result

    @staticmethod
    def fill_missing(
        data: list[dict[str, Any]],
        fill_value: Any = None,
    ) -> list[dict[str, Any]]:
        """
        Fill missing values.

        Args:
            data: Input data
            fill_value: Value to fill with

        Returns:
            Transformed data
        """
        result = []

        for row in data:
            new_row = {}
            for key, value in row.items():
                new_row[key] = value if value not in (None, "", "NA", "null") else fill_value
            result.append(new_row)

        return result

    @staticmethod
    def select_columns(
        data: list[dict[str, Any]],
        columns: list[str],
    ) -> list[dict[str, Any]]:
        """
        Select specific columns.

        Args:
            data: Input data
            columns: Columns to keep

        Returns:
            Transformed data
        """
        result = []

        for row in data:
            new_row = {col: row.get(col) for col in columns}
            result.append(new_row)

        return result


class DataProfiler:
    """Generates data profiles."""

    @staticmethod
    def profile(data: list[dict[str, Any]], sample_size: int = 5) -> DataProfile:
        """
        Generate data profile.

        Args:
            data: Input data
            sample_size: Number of sample rows

        Returns:
            DataProfile
        """
        if not data:
            return DataProfile(
                row_count=0,
                column_count=0,
                columns={},
                missing_values={},
                sample_rows=[],
            )

        # Get columns
        columns = list(data[0].keys())
        column_count = len(columns)

        # Analyze columns
        column_stats: dict[str, dict[str, Any]] = {}

        for col in columns:
            values = [row.get(col) for row in data]
            non_null = [v for v in values if v not in (None, "", "NA")]

            column_stats[col] = {
                "type": type(non_null[0]).__name__ if non_null else "null",
                "null_count": len(values) - len(non_null),
                "unique_count": len(set(str(v) for v in non_null)),
            }

        # Count missing values
        missing: dict[str, int] = {}

        for col in columns:
            count = sum(
                1 for row in data if row.get(col) in (None, "", "NA", "null")
            )
            missing[col] = count

        # Get sample rows
        sample = data[:sample_size]

        return DataProfile(
            row_count=len(data),
            column_count=column_count,
            columns=column_stats,
            missing_values=missing,
            sample_rows=sample,
        )


class DataPipeline:
    """Executes data pipelines."""

    def __init__(self):
        """Initialize pipeline."""
        self.steps: list[Callable] = []

    def add_step(self, step: Callable) -> DataPipeline:
        """
        Add a pipeline step.

        Args:
            step: Step function

        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self

    def execute(self, data: Any) -> Any:
        """
        Execute pipeline.

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        result = data

        for step in self.steps:
            result = step(result)

        return result
