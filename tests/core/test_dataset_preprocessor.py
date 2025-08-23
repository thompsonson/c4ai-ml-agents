"""Comprehensive tests for DatasetPreprocessor and NumpyJSONEncoder."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ml_agents.core.dataset_preprocessor import DatasetPreprocessor, NumpyJSONEncoder


class TestNumpyJSONEncoder:
    """Test the custom JSON encoder that handles numpy types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = NumpyJSONEncoder()

    def test_encode_numpy_array(self):
        """Test encoding numpy arrays to lists."""
        # 1D array
        arr_1d = np.array([1, 2, 3, 4])
        result = json.dumps(arr_1d, cls=NumpyJSONEncoder)
        assert result == "[1, 2, 3, 4]"

        # 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        result = json.dumps(arr_2d, cls=NumpyJSONEncoder)
        assert result == "[[1, 2], [3, 4]]"

        # Empty array
        arr_empty = np.array([])
        result = json.dumps(arr_empty, cls=NumpyJSONEncoder)
        assert result == "[]"

    def test_encode_numpy_scalars(self):
        """Test encoding numpy scalar types."""
        # Integer types
        int32_val = np.int32(42)
        result = json.dumps(int32_val, cls=NumpyJSONEncoder)
        assert result == "42"

        int64_val = np.int64(123456789)
        result = json.dumps(int64_val, cls=NumpyJSONEncoder)
        assert result == "123456789"

        # Float types
        float32_val = np.float32(3.14)
        result = json.dumps(float32_val, cls=NumpyJSONEncoder)
        assert abs(json.loads(result) - 3.14) < 0.01  # Account for float precision

        float64_val = np.float64(2.718281828)
        result = json.dumps(float64_val, cls=NumpyJSONEncoder)
        assert abs(json.loads(result) - 2.718281828) < 0.000001

    def test_encode_numpy_boolean(self):
        """Test encoding numpy boolean types."""
        bool_true = np.bool_(True)
        result = json.dumps(bool_true, cls=NumpyJSONEncoder)
        assert result == "true"

        bool_false = np.bool_(False)
        result = json.dumps(bool_false, cls=NumpyJSONEncoder)
        assert result == "false"

    def test_encode_pandas_nan(self):
        """Test that our encoder handles NaN detection properly."""
        # The key insight: pd.isna() detection happens in our encoder's default method
        # which is only called for objects that aren't natively JSON serializable.
        #
        # NaN values are natively JSON serializable (they become "NaN" string),
        # so our encoder won't be called for pure NaN values.
        #
        # However, our encoder is useful for preprocessing pandas DataFrames
        # where we might encounter various NaN-like objects that need conversion.

        # Test that our encoder doesn't break normal NaN handling
        nan_value = np.nan
        test_data = {"value": nan_value, "normal": 42}
        result = json.dumps(test_data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)

        # NaN remains NaN (standard JSON behavior)
        import math

        assert math.isnan(parsed["value"])
        assert parsed["normal"] == 42

        # Test that our pd.isna() check works for custom objects
        # Create a mock object that would trigger our encoder
        class CustomNaNLike:
            def __str__(self):
                return "NaN-like"

        custom_obj = CustomNaNLike()

        # Mock pd.isna to return True for our custom object
        with patch("pandas.isna", return_value=True):
            # This should trigger our encoder's default method
            result = json.dumps(custom_obj, cls=NumpyJSONEncoder)
            assert result == "null"

    def test_encode_mixed_numpy_data(self):
        """Test encoding complex data structures with mixed numpy types."""
        complex_data = {
            "array_1d": np.array([1, 2, 3]),
            "array_2d": np.array([[1, 2], [3, 4]]),
            "scalar_int": np.int32(42),
            "scalar_float": np.float64(3.14159),
            "scalar_bool": np.bool_(True),
            "regular_data": "hello world",
            "nested": {
                "inner_array": np.array([5, 6, 7]),
                "inner_scalar": np.int64(999),
            },
        }

        result = json.dumps(complex_data, cls=NumpyJSONEncoder, indent=2)
        parsed = json.loads(result)

        # Verify conversions
        assert parsed["array_1d"] == [1, 2, 3]
        assert parsed["array_2d"] == [[1, 2], [3, 4]]
        assert parsed["scalar_int"] == 42
        assert abs(parsed["scalar_float"] - 3.14159) < 0.00001
        assert parsed["scalar_bool"] is True
        assert parsed["regular_data"] == "hello world"
        assert parsed["nested"]["inner_array"] == [5, 6, 7]
        assert parsed["nested"]["inner_scalar"] == 999

    def test_encode_dataset_schema_like_structure(self):
        """Test encoding structures similar to dataset schema info."""
        # Simulate the kind of data that caused the original error
        schema_info = {
            "dataset_name": "test-dataset",
            "total_samples": 1000,
            "columns": ["input", "output"],
            "column_types": {
                "input": "object",
                "output": "int64",  # This could be a numpy type
            },
            "sample_data": [
                {
                    "input": "What is 2+2?",
                    "output": np.int64(4),  # Numpy scalar
                    "metadata": np.array([1, 2, 3]),  # Numpy array
                },
                {
                    "input": "What is the capital of France?",
                    "output": np.int64(1),
                    "metadata": np.array([4, 5, 6]),
                },
            ],
        }

        # This should not raise a JSON serialization error
        result = json.dumps(schema_info, cls=NumpyJSONEncoder, indent=2)
        parsed = json.loads(result)

        # Verify the structure is preserved and numpy types converted
        assert parsed["dataset_name"] == "test-dataset"
        assert parsed["total_samples"] == 1000
        assert len(parsed["sample_data"]) == 2
        assert parsed["sample_data"][0]["output"] == 4
        assert parsed["sample_data"][0]["metadata"] == [1, 2, 3]
        assert parsed["sample_data"][1]["metadata"] == [4, 5, 6]

    def test_encode_unsupported_types_fallback(self):
        """Test that unsupported types fall back to default JSON encoder behavior."""

        # Test with a custom object that should raise TypeError
        class CustomObject:
            def __init__(self, value):
                self.value = value

        custom_obj = CustomObject(42)

        with pytest.raises(TypeError):
            json.dumps(custom_obj, cls=NumpyJSONEncoder)

    def test_roundtrip_consistency(self):
        """Test that data can be encoded and decoded consistently."""
        original_data = {
            "numbers": np.array([1, 2, 3, 4, 5]),
            "floats": np.array([1.1, 2.2, 3.3]),
            "booleans": [np.bool_(True), np.bool_(False)],
            "scalars": {
                "int": np.int32(42),
                "float": np.float64(3.14),
                "bool": np.bool_(True),
            },
        }

        # Encode and decode
        encoded = json.dumps(original_data, cls=NumpyJSONEncoder)
        decoded = json.loads(encoded)

        # Verify the data matches (allowing for type conversions)
        assert decoded["numbers"] == [1, 2, 3, 4, 5]
        assert decoded["floats"] == [1.1, 2.2, 3.3]
        assert decoded["booleans"] == [True, False]
        assert decoded["scalars"]["int"] == 42
        assert abs(decoded["scalars"]["float"] - 3.14) < 0.00001
        assert decoded["scalars"]["bool"] is True


class TestDatasetPreprocessorCore:
    """Test core DatasetPreprocessor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DatasetPreprocessor()

    def test_initialization(self):
        """Test DatasetPreprocessor initialization."""
        # Test without parameters
        preprocessor = DatasetPreprocessor()
        assert preprocessor.benchmark_csv is None
        assert preprocessor.db_config is None
        assert preprocessor.db_manager is None

    def test_initialization_with_params(self):
        """Test DatasetPreprocessor initialization with parameters."""
        with patch("ml_agents.core.dataset_preprocessor.DatabaseConfig"):
            with patch("ml_agents.core.dataset_preprocessor.DatabaseManager"):
                preprocessor = DatasetPreprocessor("test.csv", "test.db")
                assert preprocessor.benchmark_csv == "test.csv"

    def test_pattern_detection_algorithms(self):
        """Test pattern detection with synthetic data."""
        # Create a test DataFrame with various patterns
        test_data = pd.DataFrame(
            {
                "question": [
                    "What is 2+2?",
                    "What is the capital of France?",
                    "What is the meaning of life?",
                ],
                "answer": ["4", "Paris", "42"],
                "context": [
                    "Math problem",
                    "Geography question",
                    "Philosophy question",
                ],
                "metadata_id": [1, 2, 3],
            }
        )

        patterns = self.preprocessor._detect_patterns(test_data)

        # Should detect question as input candidate
        input_candidates = patterns["single_field_input_candidates"]
        assert len(input_candidates) > 0
        assert any(candidate[0] == "question" for candidate in input_candidates)

        # Should detect answer as output candidate
        output_candidates = patterns["single_field_output_candidates"]
        assert len(output_candidates) > 0
        assert any(candidate[0] == "answer" for candidate in output_candidates)

    def test_confidence_calculation(self):
        """Test confidence score calculation for pattern matching."""
        # Test exact match
        confidence = self.preprocessor._calculate_pattern_confidence(
            "question", "question"
        )
        assert confidence == 1.0

        # Test partial match
        confidence = self.preprocessor._calculate_pattern_confidence(
            "user_question", "question"
        )
        assert 0.5 < confidence < 1.0

        # Test no match
        confidence = self.preprocessor._calculate_pattern_confidence(
            "random_field", "question"
        )
        assert confidence < 0.5

    def test_completeness_scoring(self):
        """Test completeness scoring for field names."""
        # Test completeness indicators
        score_full = self.preprocessor._get_completeness_score("full_answer")
        score_simple = self.preprocessor._get_completeness_score("answer")
        assert score_full > score_simple

        # Test oracle/ground truth indicators
        score_oracle = self.preprocessor._get_completeness_score("oracle_answer")
        score_regular = self.preprocessor._get_completeness_score("answer")
        assert score_oracle > score_regular

    def test_content_analysis(self):
        """Test content type analysis."""
        # Test numeric content
        numeric_series = pd.Series([1, 2, 3, 4, 5])
        analysis = self.preprocessor._analyze_content_type(numeric_series)
        assert analysis["type"] == "numeric"

        # Test text content
        text_series = pd.Series(["hello", "world", "test", "content"])
        analysis = self.preprocessor._analyze_content_type(text_series)
        assert analysis["type"] == "text"
        assert analysis["avg_length"] > 0

        # Test boolean content
        bool_series = pd.Series(["true", "false", "true", "false"])
        analysis = self.preprocessor._analyze_content_type(bool_series)
        assert analysis["type"] == "boolean"

    def test_transformation_rule_generation(self):
        """Test transformation rule generation."""
        mock_schema = {
            "dataset_name": "test-dataset",
            "detected_patterns": {
                "recommended_pattern": {
                    "type": "single_field",
                    "confidence": 0.9,
                    "input_fields": ["question"],
                    "output_field": "answer",
                }
            },
        }

        rules = self.preprocessor.generate_transformation_rules(mock_schema)

        assert rules["dataset_name"] == "test-dataset"
        assert rules["transformation_type"] == "single_field"
        assert rules["confidence"] == 0.9
        assert rules["input_fields"] == ["question"]
        assert rules["output_field"] == "answer"

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    @patch("ml_agents.core.dataset_preprocessor.get_dataset_config_info")
    def test_json_serialization_integration(self, mock_config_info, mock_load_dataset):
        """Test that schema inspection can be JSON serialized without errors."""
        # Mock dataset with numpy data
        mock_df = pd.DataFrame(
            {
                "question": ["What is 2+2?", "What is 3+3?"],
                "answer": [np.int64(4), np.int64(6)],  # Numpy integers
                "scores": [np.array([1, 2, 3]), np.array([4, 5, 6])],  # Numpy arrays
            }
        )

        mock_dataset = Mock()
        mock_dataset.select.return_value.to_pandas.return_value = mock_df
        mock_dataset.__len__.return_value = 100
        mock_load_dataset.return_value = mock_dataset

        mock_info = Mock()
        mock_info.splits = {"train": Mock()}
        mock_config_info.return_value = mock_info

        # This should not raise a JSON serialization error
        schema_info = self.preprocessor.inspect_dataset_schema(
            "test-dataset", sample_size=10
        )

        # Test that it can be serialized with our custom encoder
        json_str = json.dumps(schema_info, cls=NumpyJSONEncoder, indent=2)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["dataset_name"] == "test-dataset"
        assert "sample_data" in parsed

    def test_export_standardized_json_with_numpy(self):
        """Test exporting datasets with numpy data to JSON format."""
        # Create mock dataset with numpy types
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.__getitem__.side_effect = [
            {"INPUT": "What is 2+2?", "OUTPUT": np.int64(4)},
            {"INPUT": "What is 3+3?", "OUTPUT": np.int64(6)},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            # This should not raise a JSON serialization error
            self.preprocessor.export_standardized(mock_dataset, output_path)

            # Verify the file was created and contains valid JSON
            with open(output_path, "r") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["INPUT"] == "What is 2+2?"
            assert data[0]["OUTPUT"] == 4  # Should be converted from numpy

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestDatasetPreprocessorDatabase:
    """Test database integration functionality."""

    def setup_method(self):
        """Set up test fixtures with mock database."""
        with patch("ml_agents.core.dataset_preprocessor.DatabaseConfig"):
            with patch(
                "ml_agents.core.dataset_preprocessor.DatabaseManager"
            ) as mock_db_manager:
                self.mock_db_manager = mock_db_manager.return_value
                self.preprocessor = DatasetPreprocessor(db_path="test.db")

    def test_save_preprocessing_metadata_with_numpy(self):
        """Test saving metadata containing numpy data to database."""
        schema_info = {
            "dataset_name": "test-dataset",
            "columns": ["input", "output"],
            "sample_data": [
                {"input": "test", "output": np.int64(42)},
                {"metadata": np.array([1, 2, 3])},
            ],
        }

        rules = {"confidence": 0.9, "transformation_type": "single_field"}

        validation_results = {
            "validation_passed": True,
            "metrics": np.array([0.8, 0.9, 1.0]),  # Numpy array in validation
        }

        # Mock database connection
        mock_conn = Mock()
        self.mock_db_manager.get_connection.return_value.__enter__.return_value = (
            mock_conn
        )

        # This should not raise a JSON serialization error
        record_id = self.preprocessor._save_preprocessing_metadata(
            dataset_name="test-dataset",
            dataset_url="test-url",
            schema_info=schema_info,
            rules=rules,
            validation_results=validation_results,
            output_path="test.json",
        )

        # Verify that execute was called (meaning JSON serialization succeeded)
        assert mock_conn.execute.called
        assert len(record_id) > 0  # Should return a valid UUID string
