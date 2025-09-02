"""Tests for multiple choice dataset preprocessing functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ml_agents.core.dataset_preprocessor import DatasetPreprocessor, NumpyJSONEncoder


class TestMultipleChoiceFormatting:
    """Test formatting of multiple choice options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DatasetPreprocessor()

    def test_format_multiple_choice_options_basic(self):
        """Test basic formatting of candidate answers."""
        candidates = [
            "the small yellow circle",
            "the medium yellow circle",
            "both of them",
            "none of them",
        ]

        result = self.preprocessor._format_multiple_choice_options(candidates)

        expected = "A) the small yellow circle\nB) the medium yellow circle\nC) both of them\nD) none of them"
        assert result == expected

    def test_format_multiple_choice_options_with_whitespace(self):
        """Test formatting handles whitespace in candidates."""
        candidates = [
            "  option with leading space",
            "option with trailing space  ",
            "  option with both  ",
            "normal option",
        ]

        result = self.preprocessor._format_multiple_choice_options(candidates)

        assert "A) option with leading space" in result
        assert "B) option with trailing space" in result
        assert "C) option with both" in result
        assert "D) normal option" in result

    def test_format_multiple_choice_options_empty_list(self):
        """Test formatting empty candidate list."""
        candidates = []

        result = self.preprocessor._format_multiple_choice_options(candidates)

        assert result == ""

    def test_format_multiple_choice_options_single_candidate(self):
        """Test formatting single candidate."""
        candidates = ["only option"]

        result = self.preprocessor._format_multiple_choice_options(candidates)

        assert result == "A) only option"

    def test_format_multiple_choice_options_many_candidates(self):
        """Test formatting more than 26 candidates (beyond Z)."""
        # Create 30 candidates to test beyond Z
        candidates = [f"option {i}" for i in range(30)]

        result = self.preprocessor._format_multiple_choice_options(candidates)

        # Check first few
        assert "A) option 0" in result
        assert "B) option 1" in result
        assert "Z) option 25" in result
        # After Z, it should continue with ASCII characters
        assert "[) option 26" in result  # ASCII 91 after Z (90)

    def test_format_multiple_choice_options_special_characters(self):
        """Test formatting candidates with special characters."""
        candidates = [
            "Option with 'quotes'",
            'Option with "double quotes"',
            "Option with $pecial ch@rs!",
            "Option with\nnewline",
        ]

        result = self.preprocessor._format_multiple_choice_options(candidates)

        assert "A) Option with 'quotes'" in result
        assert 'B) Option with "double quotes"' in result
        assert "C) Option with $pecial ch@rs!" in result
        assert "D) Option with\nnewline" in result


class TestStoryQuestionChoicesPattern:
    """Test story_question_choices pattern detection and processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DatasetPreprocessor()

    def test_detect_story_question_choices_pattern(self):
        """Test detection of story_question_choices pattern."""
        # Create test DataFrame with story, question, candidate_answers columns
        test_data = pd.DataFrame(
            {
                "story": ["Once upon a time...", "In a distant galaxy..."],
                "question": ["What happened next?", "Who was the hero?"],
                "answer": [0, 1],
                "candidate_answers": [
                    ["option A", "option B", "option C"],
                    ["hero 1", "hero 2", "hero 3"],
                ],
            }
        )

        patterns = self.preprocessor._detect_patterns(test_data)

        # Check that story_question_choices pattern is detected
        multi_patterns = patterns.get("multi_field_patterns", {})
        assert "story_question_choices" in multi_patterns
        assert "story" in multi_patterns["story_question_choices"]
        assert "question" in multi_patterns["story_question_choices"]
        assert "candidate_answers" in multi_patterns["story_question_choices"]

    def test_generate_rules_for_story_question_choices(self):
        """Test rule generation for story_question_choices pattern."""
        mock_schema = {
            "dataset_name": "test-multichoice",
            "detected_patterns": {
                "recommended_pattern": {
                    "type": "multi_field",
                    "pattern_name": "story_question_choices",
                    "confidence": 0.95,
                    "input_fields": ["story", "question", "candidate_answers"],
                    "output_field": "answer",
                }
            },
        }

        rules = self.preprocessor.generate_transformation_rules(mock_schema)

        # Check generated rules
        assert rules["transformation_type"] == "multi_field"
        assert rules["input_format"] == "multi_field"
        assert rules["input_fields"] == ["story", "question", "candidate_answers"]
        assert rules["output_field"] == "answer"
        assert "resolve_answer_index" in rules["preprocessing_steps"]

        # Check field labels
        assert rules["field_labels"]["story"] == "STORY:"
        assert rules["field_labels"]["question"] == "QUESTION:"
        assert rules["field_labels"]["candidate_answers"] == "OPTIONS:"

    def test_pattern_recommendation_prefers_story_question_choices(self):
        """Test that story_question_choices pattern is preferred when detected."""
        # Create mock data with both single field and multi-field patterns
        input_candidates = [("question", "question", 1.0)]
        output_candidates = [("answer", "answer", 1.0)]
        multi_field_matches = {
            "story_question_choices": [
                "story",
                "question",
                "candidate_answers",
                "answer",
            ]
        }

        recommendation = self.preprocessor._recommend_pattern(
            input_candidates, output_candidates, multi_field_matches
        )

        assert recommendation["type"] == "multi_field"
        assert recommendation["pattern_name"] == "story_question_choices"
        assert "story" in recommendation["input_fields"]
        assert "question" in recommendation["input_fields"]
        assert "candidate_answers" in recommendation["input_fields"]


class TestAnswerIndexResolution:
    """Test answer index to text resolution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DatasetPreprocessor()

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_index_valid(self, mock_load_dataset):
        """Test resolving valid answer index to text."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["story", "question", "answer", "candidate_answers"]
        mock_dataset.__len__ = Mock(return_value=1)

        # Mock the map function to test transformation
        def mock_map(transform_func, **kwargs):
            example = {
                "story": "Test story",
                "question": "Test question?",
                "answer": 1,  # Index 1
                "candidate_answers": ["option A", "option B", "option C"],
            }
            result = transform_func(example)

            # Create a mock dataset with the transformed result
            transformed = Mock()
            transformed.column_names = ["INPUT", "OUTPUT"]
            transformed.__len__ = Mock(return_value=1)
            transformed.__getitem__ = Mock(return_value=result)
            return transformed

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        # Create transformation rules with resolve_answer_index
        rules = {
            "dataset_name": "test",
            "input_format": "multi_field",
            "input_fields": ["story", "question", "candidate_answers"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "story": "STORY:",
                "question": "QUESTION:",
                "candidate_answers": "OPTIONS:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
        }

        # Apply transformation
        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

        # Check that answer index was resolved to text
        result = transformed[0]
        assert result["OUTPUT"] == "option B"  # Index 1 should resolve to "option B"

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_index_out_of_bounds(self, mock_load_dataset):
        """Test handling of out-of-bounds answer index."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["story", "question", "answer", "candidate_answers"]
        mock_dataset.__len__ = Mock(return_value=1)

        def mock_map(transform_func, **kwargs):
            example = {
                "story": "Test story",
                "question": "Test question?",
                "answer": 5,  # Out of bounds index
                "candidate_answers": ["option A", "option B", "option C"],
            }
            result = transform_func(example)

            transformed = Mock()
            transformed.column_names = ["INPUT", "OUTPUT"]
            transformed.__len__ = Mock(return_value=1)
            transformed.__getitem__ = Mock(return_value=result)
            return transformed

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        rules = {
            "dataset_name": "test",
            "input_format": "multi_field",
            "input_fields": ["story", "question", "candidate_answers"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "story": "STORY:",
                "question": "QUESTION:",
                "candidate_answers": "OPTIONS:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
        }

        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

        # Should keep the original value when index is out of bounds
        result = transformed[0]
        assert result["OUTPUT"] == "5"

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_no_resolve_when_not_in_preprocessing_steps(self, mock_load_dataset):
        """Test that answer index is not resolved when not in preprocessing_steps."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["story", "question", "answer", "candidate_answers"]
        mock_dataset.__len__ = Mock(return_value=1)

        def mock_map(transform_func, **kwargs):
            example = {
                "story": "Test story",
                "question": "Test question?",
                "answer": 1,
                "candidate_answers": ["option A", "option B", "option C"],
            }
            result = transform_func(example)

            transformed = Mock()
            transformed.column_names = ["INPUT", "OUTPUT"]
            transformed.__len__ = Mock(return_value=1)
            transformed.__getitem__ = Mock(return_value=result)
            return transformed

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        rules = {
            "dataset_name": "test",
            "input_format": "single_field",
            "input_fields": ["question"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {},
            "preprocessing_steps": [],  # No resolve_answer_index
        }

        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

        # Should keep the numeric index
        result = transformed[0]
        assert result["OUTPUT"] == "1"


class TestMultipleChoiceIntegration:
    """Integration tests for complete multiple choice dataset processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DatasetPreprocessor()

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    @patch("datasets.get_dataset_config_info")
    def test_end_to_end_multiple_choice_transformation(
        self, mock_config_info, mock_load_dataset
    ):
        """Test complete transformation of a multiple choice dataset."""
        # Create realistic test data
        test_df = pd.DataFrame(
            {
                "story": [
                    "We have three blocks, A, B and C. Block B is below C.",
                    "In a room, there are four objects: a ball, a cube, a pyramid, and a cylinder.",
                ],
                "question": [
                    "Which block is below C?",
                    "How many objects are in the room?",
                ],
                "answer": [0, 3],  # Indices into candidate_answers
                "candidate_answers": [
                    ["Block B", "Block A", "Block C", "None"],
                    ["Two", "Three", "Four", "Five"],
                ],
            }
        )

        # Mock dataset loading
        mock_dataset = Mock()
        mock_dataset.select.return_value.to_pandas.return_value = test_df
        mock_dataset.__len__ = Mock(return_value=len(test_df))
        mock_dataset.column_names = list(test_df.columns)

        # Mock the map function for transformation
        def mock_map(transform_func, **kwargs):
            transformed_data = []
            for idx, row in test_df.iterrows():
                result = transform_func(row.to_dict())
                transformed_data.append(result)

            transformed = Mock()
            transformed.column_names = ["INPUT", "OUTPUT"]
            transformed.__len__ = Mock(return_value=len(transformed_data))
            transformed.__getitem__ = Mock(side_effect=lambda i: transformed_data[i])
            transformed.__iter__ = Mock(return_value=iter(transformed_data))

            # Add to_pandas method for validation
            transformed_df = pd.DataFrame(transformed_data)
            transformed.to_pandas.return_value = transformed_df

            return transformed

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        # Mock config info
        mock_info = Mock()
        mock_info.splits = {"train": Mock()}
        mock_config_info.return_value = mock_info

        # Use specific transformation rules for story_question_choices pattern
        # (instead of relying on automatic detection which can be flaky with mocks)
        rules = {
            "dataset_name": "test-dataset",
            "transformation_type": "multi_field",
            "confidence": 1.0,
            "input_format": "multi_field",
            "input_fields": ["story", "question", "candidate_answers"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "story": "STORY:",
                "question": "QUESTION:",
                "candidate_answers": "OPTIONS:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
        }

        # Apply transformation
        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

        # Verify first sample
        first_sample = transformed[0]
        assert "STORY:" in first_sample["INPUT"]
        assert "We have three blocks" in first_sample["INPUT"]
        assert "QUESTION:" in first_sample["INPUT"]
        assert "Which block is below C?" in first_sample["INPUT"]
        assert "OPTIONS:" in first_sample["INPUT"]
        assert "A) Block B" in first_sample["INPUT"]
        assert "B) Block A" in first_sample["INPUT"]
        assert "C) Block C" in first_sample["INPUT"]
        assert "D) None" in first_sample["INPUT"]
        assert first_sample["OUTPUT"] == "Block B"  # Resolved from index 0

        # Verify second sample
        second_sample = transformed[1]
        assert "four objects" in second_sample["INPUT"]
        assert "How many objects" in second_sample["INPUT"]
        assert "A) Two" in second_sample["INPUT"]
        assert "D) Five" in second_sample["INPUT"]
        assert second_sample["OUTPUT"] == "Five"  # Resolved from index 3

    def test_export_multiple_choice_dataset(self):
        """Test exporting a transformed multiple choice dataset."""
        # Create mock transformed dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=2)

        def getitem_func(i):
            items = [
                {
                    "INPUT": "STORY:\n\nTest story\n\nQUESTION:\n\nTest question?\n\nOPTIONS:\n\nA) option 1\nB) option 2",
                    "OUTPUT": "option 1",
                },
                {
                    "INPUT": "STORY:\n\nAnother story\n\nQUESTION:\n\nAnother question?\n\nOPTIONS:\n\nA) choice A\nB) choice B",
                    "OUTPUT": "choice B",
                },
            ]
            return items[i] if 0 <= i < len(items) else None

        mock_dataset.__getitem__ = Mock(side_effect=getitem_func)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            # Export to JSON
            self.preprocessor.export_standardized(mock_dataset, output_path)

            # Verify the exported file
            with open(output_path, "r") as f:
                data = json.load(f)

            assert len(data) == 2
            assert "STORY:" in data[0]["INPUT"]
            assert "QUESTION:" in data[0]["INPUT"]
            assert "OPTIONS:" in data[0]["INPUT"]
            assert data[0]["OUTPUT"] == "option 1"
            assert data[1]["OUTPUT"] == "choice B"

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DatasetPreprocessor()

    def test_format_options_with_none_values(self):
        """Test formatting candidates with None values."""
        candidates = ["option 1", None, "option 3", None]

        result = self.preprocessor._format_multiple_choice_options(candidates)

        assert "A) option 1" in result
        assert "B) None" in result  # None should be converted to string
        assert "C) option 3" in result
        assert "D) None" in result

    def test_format_options_with_numeric_values(self):
        """Test formatting candidates with numeric values."""
        candidates = [1, 2.5, "three", 4]

        result = self.preprocessor._format_multiple_choice_options(candidates)

        assert "A) 1" in result
        assert "B) 2.5" in result
        assert "C) three" in result
        assert "D) 4" in result

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_with_non_list_candidates(self, mock_load_dataset):
        """Test answer resolution when candidate_answers is not a list."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["story", "question", "answer", "candidate_answers"]
        mock_dataset.__len__ = Mock(return_value=1)

        def mock_map(transform_func, **kwargs):
            example = {
                "story": "Test story",
                "question": "Test question?",
                "answer": 0,
                "candidate_answers": "not a list",  # Invalid format
            }
            result = transform_func(example)

            transformed = Mock()
            transformed.column_names = ["INPUT", "OUTPUT"]
            transformed.__len__ = Mock(return_value=1)
            transformed.__getitem__ = Mock(return_value=result)
            return transformed

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        rules = {
            "dataset_name": "test",
            "input_format": "multi_field",
            "input_fields": ["story", "question", "candidate_answers"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "story": "STORY:",
                "question": "QUESTION:",
                "candidate_answers": "OPTIONS:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
        }

        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

        # Should keep the original value when candidates is not a list
        result = transformed[0]
        assert result["OUTPUT"] == "0"
        # Input should still include the candidates field as string
        assert "not a list" in result["INPUT"]

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_with_negative_index(self, mock_load_dataset):
        """Test answer resolution with negative index."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["story", "question", "answer", "candidate_answers"]
        mock_dataset.__len__ = Mock(return_value=1)

        def mock_map(transform_func, **kwargs):
            example = {
                "story": "Test story",
                "question": "Test question?",
                "answer": -1,  # Negative index (Python-style)
                "candidate_answers": ["option A", "option B", "option C"],
            }
            result = transform_func(example)

            transformed = Mock()
            transformed.column_names = ["INPUT", "OUTPUT"]
            transformed.__len__ = Mock(return_value=1)
            transformed.__getitem__ = Mock(return_value=result)
            return transformed

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        rules = {
            "dataset_name": "test",
            "input_format": "multi_field",
            "input_fields": ["story", "question", "candidate_answers"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "story": "STORY:",
                "question": "QUESTION:",
                "candidate_answers": "OPTIONS:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
        }

        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

        # Negative indices are not supported, should keep original value
        result = transformed[0]
        assert result["OUTPUT"] == "-1"

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_index_with_options_field(self, mock_load_dataset):
        """Test resolving answer index with 'options' field (like GPQA dataset)."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "options", "answer"]
        mock_dataset.__len__ = Mock(return_value=1)

        def mock_map(transform_func, **kwargs):
            example = {
                "question": "Test question?",
                "options": ["option A", "option B", "option C", "option D"],
                "answer": 2,  # Index 2 should be "option C"
            }
            result = transform_func(example)

            # Check if the answer was resolved correctly
            assert result["OUTPUT"] == "option C"
            return Mock(column_names=["INPUT", "OUTPUT"], __len__=Mock(return_value=1))

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        # Rules without specifying answer_options_field (should auto-detect)
        rules = {
            "dataset_name": "test",
            "input_format": "multi_field",
            "input_fields": ["question", "options"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "question": "QUESTION:",
                "options": "OPTIONS:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
        }

        transformed = self.preprocessor.apply_transformation("test-dataset", rules)
        # Test passes if no exception and assertion in mock_map passes

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_index_with_explicit_field_specification(
        self, mock_load_dataset
    ):
        """Test using explicit answer_options_field in rules."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "my_choices", "answer"]
        mock_dataset.__len__ = Mock(return_value=1)

        def mock_map(transform_func, **kwargs):
            example = {
                "question": "Test question?",
                "my_choices": ["choice 1", "choice 2", "choice 3"],
                "answer": 1,
            }
            result = transform_func(example)

            # Should resolve to "choice 2" (index 1)
            assert result["OUTPUT"] == "choice 2"
            return Mock(column_names=["INPUT", "OUTPUT"], __len__=Mock(return_value=1))

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        # Explicitly specify the answer_options_field
        rules = {
            "dataset_name": "test",
            "input_format": "multi_field",
            "input_fields": ["question", "my_choices"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "question": "QUESTION:",
                "my_choices": "CHOICES:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
            "answer_options_field": "my_choices",  # Explicitly specify
        }

        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_index_auto_detect_various_fields(self, mock_load_dataset):
        """Test auto-detection of various common answer option field names."""
        test_cases = [
            ("choices", ["A", "B", "C"]),
            ("alternatives", ["Alt 1", "Alt 2", "Alt 3"]),
            ("answers", ["ans1", "ans2", "ans3"]),
            ("answer_choices", ["choice A", "choice B", "choice C"]),
        ]

        for field_name, options_list in test_cases:
            mock_dataset = Mock()
            mock_dataset.column_names = ["question", field_name, "answer"]
            mock_dataset.__len__ = Mock(return_value=1)

            def create_mock_map(expected_field, expected_options):
                def mock_map(transform_func, **kwargs):
                    example = {
                        "question": "Test question?",
                        expected_field: expected_options,
                        "answer": 1,
                    }
                    result = transform_func(example)
                    # Should auto-detect and resolve index 1
                    assert result["OUTPUT"] == expected_options[1]
                    return Mock(
                        column_names=["INPUT", "OUTPUT"], __len__=Mock(return_value=1)
                    )

                return mock_map

            mock_dataset.map = create_mock_map(field_name, options_list)
            mock_load_dataset.return_value = mock_dataset

            rules = {
                "dataset_name": "test",
                "input_format": "multi_field",
                "input_fields": ["question", field_name],
                "output_field": "answer",
                "field_separator": "\n\n",
                "field_labels": {
                    "question": "QUESTION:",
                    field_name: "OPTIONS:",
                },
                "preprocessing_steps": ["resolve_answer_index"],
                # No answer_options_field specified - should auto-detect
            }

            transformed = self.preprocessor.apply_transformation("test-dataset", rules)

    @patch("ml_agents.core.dataset_preprocessor.load_dataset")
    def test_resolve_answer_index_fallback_to_any_list_field(self, mock_load_dataset):
        """Test fallback to any list field when no common names found."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "unusual_field_name", "answer"]
        mock_dataset.__len__ = Mock(return_value=1)

        def mock_map(transform_func, **kwargs):
            example = {
                "question": "Test question?",
                "unusual_field_name": ["item1", "item2", "item3", "item4"],
                "answer": 3,
            }
            result = transform_func(example)

            # Should detect unusual_field_name as the list field and resolve index 3
            assert result["OUTPUT"] == "item4"
            return Mock(column_names=["INPUT", "OUTPUT"], __len__=Mock(return_value=1))

        mock_dataset.map = mock_map
        mock_load_dataset.return_value = mock_dataset

        rules = {
            "dataset_name": "test",
            "input_format": "multi_field",
            "input_fields": ["question", "unusual_field_name"],
            "output_field": "answer",
            "field_separator": "\n\n",
            "field_labels": {
                "question": "QUESTION:",
                "unusual_field_name": "OPTIONS:",
            },
            "preprocessing_steps": ["resolve_answer_index"],
        }

        transformed = self.preprocessor.apply_transformation("test-dataset", rules)

    def test_validation_with_multiple_choice_dataset(self):
        """Test validation of transformed multiple choice dataset."""
        # Create mock original dataset
        original = Mock()
        original.__len__ = Mock(return_value=10)
        original.to_pandas.return_value = pd.DataFrame(
            {
                "story": ["story"] * 10,
                "question": ["question"] * 10,
                "answer": list(range(10)),
                "candidate_answers": [["a", "b", "c"]] * 10,
            }
        )

        # Create mock transformed dataset
        transformed = Mock()
        transformed.__len__ = Mock(return_value=10)
        transformed_data = []
        for i in range(10):
            transformed_data.append(
                {
                    "INPUT": f"STORY:\n\nstory\n\nQUESTION:\n\nquestion\n\nOPTIONS:\n\nA) a\nB) b\nC) c",
                    "OUTPUT": ["a", "b", "c"][i % 3],
                }
            )
        transformed.to_pandas.return_value = pd.DataFrame(transformed_data)
        transformed.__iter__ = Mock(return_value=iter(transformed_data))

        # Run validation
        validation_results = self.preprocessor.validate_transformation(
            original, transformed
        )

        assert validation_results["sample_count_preserved"] == True
        assert validation_results["original_samples"] == 10
        assert validation_results["transformed_samples"] == 10
        assert validation_results["empty_inputs"] == 0
        assert validation_results["empty_outputs"] == 0
