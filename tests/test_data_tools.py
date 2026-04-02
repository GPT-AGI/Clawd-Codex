"""Tests for data_tools module."""

import unittest
import json
from pathlib import Path
import tempfile
import shutil

from src.data_tools import (
    DataProfile,
    DataReader,
    DataWriter,
    DataTransformer,
    DataProfiler,
    DataPipeline,
)


class TestDataTools(unittest.TestCase):
    """Test cases for data tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)

    def test_read_csv(self):
        """Test reading CSV file."""
        csv_path = self.test_dir / "test.csv"
        csv_path.write_text("name,age\nAlice,30\nBob,25\n")

        data = DataReader.read_csv(csv_path)

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["name"], "Alice")

    def test_read_json(self):
        """Test reading JSON file."""
        json_path = self.test_dir / "test.json"
        json_path.write_text('{"key": "value"}')

        data = DataReader.read_json(json_path)

        self.assertEqual(data, {"key": "value"})

    def test_read_jsonl(self):
        """Test reading JSON Lines file."""
        jsonl_path = self.test_dir / "test.jsonl"
        jsonl_path.write_text('{"id": 1}\n{"id": 2}\n')

        data = DataReader.read_jsonl(jsonl_path)

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["id"], 1)

    def test_write_csv(self):
        """Test writing CSV file."""
        data = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
        ]

        csv_path = self.test_dir / "output.csv"
        DataWriter.write_csv(data, csv_path)

        content = csv_path.read_text()
        self.assertIn("name,age", content)
        self.assertIn("Alice", content)

    def test_write_json(self):
        """Test writing JSON file."""
        data = {"key": "value"}

        json_path = self.test_dir / "output.json"
        DataWriter.write_json(data, json_path)

        content = json.loads(json_path.read_text())
        self.assertEqual(content, data)

    def test_write_jsonl(self):
        """Test writing JSON Lines file."""
        data = [{"id": 1}, {"id": 2}]

        jsonl_path = self.test_dir / "output.jsonl"
        DataWriter.write_jsonl(data, jsonl_path)

        lines = jsonl_path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)

    def test_transform_map_columns(self):
        """Test column mapping."""
        data = [{"old_name": "value"}]
        result = DataTransformer.map_columns(data, {"old_name": "new_name"})

        self.assertIn("new_name", result[0])
        self.assertNotIn("old_name", result[0])

    def test_transform_filter_rows(self):
        """Test row filtering."""
        data = [
            {"id": 1, "active": True},
            {"id": 2, "active": False},
        ]

        result = DataTransformer.filter_rows(data, lambda r: r["active"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], 1)

    def test_transform_transform_values(self):
        """Test value transformation."""
        data = [{"value": "10"}]
        result = DataTransformer.transform_values(data, "value", int)

        self.assertEqual(result[0]["value"], 10)

    def test_transform_fill_missing(self):
        """Test filling missing values."""
        data = [{"a": None, "b": "value"}]
        result = DataTransformer.fill_missing(data, fill_value="N/A")

        self.assertEqual(result[0]["a"], "N/A")
        self.assertEqual(result[0]["b"], "value")

    def test_transform_select_columns(self):
        """Test column selection."""
        data = [{"a": 1, "b": 2, "c": 3}]
        result = DataTransformer.select_columns(data, ["a", "b"])

        self.assertIn("a", result[0])
        self.assertIn("b", result[0])
        self.assertNotIn("c", result[0])

    def test_profile_data(self):
        """Test data profiling."""
        data = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
            {"name": None, "age": "35"},
        ]

        profile = DataProfiler.profile(data)

        self.assertEqual(profile.row_count, 3)
        self.assertEqual(profile.column_count, 2)
        self.assertIn("name", profile.columns)
        self.assertIn("age", profile.columns)

    def test_profile_empty_data(self):
        """Test profiling empty data."""
        profile = DataProfiler.profile([])

        self.assertEqual(profile.row_count, 0)
        self.assertEqual(profile.column_count, 0)

    def test_pipeline_execution(self):
        """Test data pipeline."""
        data = [{"value": "10"}, {"value": "20"}]

        pipeline = DataPipeline()
        pipeline.add_step(lambda d: DataTransformer.transform_values(d, "value", int))
        pipeline.add_step(lambda d: DataTransformer.filter_rows(d, lambda r: r["value"] > 15))

        result = pipeline.execute(data)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], 20)

    def test_pipeline_chaining(self):
        """Test pipeline method chaining."""
        pipeline = DataPipeline().add_step(lambda x: x).add_step(lambda x: x)

        self.assertEqual(len(pipeline.steps), 2)


if __name__ == "__main__":
    unittest.main()
