"""Tests for answer extraction Pydantic models."""

import pytest
from pydantic import ValidationError

from src.utils.answer_extraction import (
    BaseAnswerExtraction,
    DefaultExtractionModel,
    ListExtraction,
    MultipleChoiceExtraction,
    NumericalExtraction,
    ReasoningChainExtraction,
    TextualExtraction,
    YesNoExtraction,
    get_extraction_model,
)


class TestBaseAnswerExtraction:
    """Test suite for BaseAnswerExtraction model."""

    def test_valid_base_extraction(self):
        """Test valid BaseAnswerExtraction creation."""
        extraction = BaseAnswerExtraction(
            final_answer="42", confidence=0.8, extraction_method="instructor"
        )

        assert extraction.final_answer == "42"
        assert extraction.confidence == 0.8
        assert extraction.extraction_method == "instructor"

    def test_empty_answer_validation(self):
        """Test validation of empty final_answer."""
        with pytest.raises(ValidationError) as exc_info:
            BaseAnswerExtraction(final_answer="", confidence=0.8)

        assert "Final answer cannot be empty" in str(exc_info.value)

    def test_confidence_bounds_validation(self):
        """Test validation of confidence bounds."""
        # Test lower bound
        with pytest.raises(ValidationError):
            BaseAnswerExtraction(final_answer="test", confidence=-0.1)

        # Test upper bound
        with pytest.raises(ValidationError):
            BaseAnswerExtraction(final_answer="test", confidence=1.1)

        # Valid bounds
        extraction = BaseAnswerExtraction(final_answer="test", confidence=0.0)
        assert extraction.confidence == 0.0

        extraction = BaseAnswerExtraction(final_answer="test", confidence=1.0)
        assert extraction.confidence == 1.0

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed from final_answer."""
        extraction = BaseAnswerExtraction(final_answer="  42  ", confidence=0.8)

        assert extraction.final_answer == "42"


class TestMultipleChoiceExtraction:
    """Test suite for MultipleChoiceExtraction model."""

    def test_valid_multiple_choice(self):
        """Test valid MultipleChoiceExtraction creation."""
        extraction = MultipleChoiceExtraction(
            final_answer="A",
            confidence=0.9,
            selected_option="A",
            option_reasoning="Option A is correct because...",
        )

        assert extraction.selected_option == "A"
        assert extraction.option_reasoning == "Option A is correct because..."

    def test_option_normalization(self):
        """Test option format normalization."""
        extraction = MultipleChoiceExtraction(
            final_answer="a",
            confidence=0.9,
            selected_option="a",
            option_reasoning="test",
        )

        assert extraction.selected_option == "A"

    def test_long_option_format(self):
        """Test longer option formats."""
        extraction = MultipleChoiceExtraction(
            final_answer="Option B",
            confidence=0.9,
            selected_option="Option B",
            option_reasoning="test",
        )

        assert extraction.selected_option == "Option B"


class TestNumericalExtraction:
    """Test suite for NumericalExtraction model."""

    def test_valid_numerical_extraction(self):
        """Test valid NumericalExtraction creation."""
        extraction = NumericalExtraction(
            final_answer="42",
            confidence=0.9,
            numerical_value=42,
            unit="meters",
            calculation_steps=["Step 1", "Step 2"],
        )

        assert extraction.numerical_value == 42
        assert extraction.unit == "meters"
        assert len(extraction.calculation_steps) == 2

    def test_float_values(self):
        """Test float numerical values."""
        extraction = NumericalExtraction(
            final_answer="3.14", confidence=0.9, numerical_value=3.14
        )

        assert extraction.numerical_value == 3.14

    def test_invalid_numerical_value(self):
        """Test validation of invalid numerical values."""
        with pytest.raises(ValidationError) as exc_info:
            NumericalExtraction(
                final_answer="not a number",
                confidence=0.9,
                numerical_value="not a number",
            )

        assert "valid number" in str(exc_info.value)


class TestTextualExtraction:
    """Test suite for TextualExtraction model."""

    def test_valid_textual_extraction(self):
        """Test valid TextualExtraction creation."""
        extraction = TextualExtraction(
            final_answer="This is the answer",
            confidence=0.8,
            answer_type="explanation",
            key_points=["Point 1", "Point 2"],
        )

        assert extraction.answer_type == "explanation"
        assert len(extraction.key_points) == 2

    def test_answer_type_normalization(self):
        """Test answer type normalization."""
        extraction = TextualExtraction(
            final_answer="Yes", confidence=0.8, answer_type="YES/NO"
        )

        assert extraction.answer_type == "yes/no"


class TestYesNoExtraction:
    """Test suite for YesNoExtraction model."""

    def test_valid_yes_no_extraction(self):
        """Test valid YesNoExtraction creation."""
        extraction = YesNoExtraction(
            final_answer="Yes",
            confidence=0.9,
            yes_no_answer=True,
            reasoning="Because the condition is met",
        )

        assert extraction.yes_no_answer is True
        assert extraction.reasoning == "Because the condition is met"


class TestListExtraction:
    """Test suite for ListExtraction model."""

    def test_valid_list_extraction(self):
        """Test valid ListExtraction creation."""
        extraction = ListExtraction(
            final_answer="A, B, C", confidence=0.8, items=["A", "B", "C"]
        )

        assert len(extraction.items) == 3
        assert extraction.item_count == 3

    def test_empty_item_removal(self):
        """Test removal of empty items."""
        extraction = ListExtraction(
            final_answer="A, , C", confidence=0.8, items=["A", "", "C", "  "]
        )

        assert len(extraction.items) == 2
        assert extraction.items == ["A", "C"]
        assert extraction.item_count == 2

    def test_empty_list_validation(self):
        """Test validation of empty lists."""
        with pytest.raises(ValidationError) as exc_info:
            ListExtraction(final_answer="Nothing", confidence=0.8, items=[])

        assert "cannot be empty" in str(exc_info.value)


class TestReasoningChainExtraction:
    """Test suite for ReasoningChainExtraction model."""

    def test_valid_reasoning_chain(self):
        """Test valid ReasoningChainExtraction creation."""
        extraction = ReasoningChainExtraction(
            final_answer="42",
            confidence=0.9,
            reasoning_steps=["Step 1", "Step 2", "Step 3"],
            intermediate_results=["Result 1", "Result 2"],
            uncertainty_flags=["Assumption made"],
        )

        assert len(extraction.reasoning_steps) == 3
        assert len(extraction.intermediate_results) == 2
        assert len(extraction.uncertainty_flags) == 1

    def test_empty_reasoning_steps_validation(self):
        """Test validation of empty reasoning steps."""
        with pytest.raises(ValidationError) as exc_info:
            ReasoningChainExtraction(
                final_answer="42", confidence=0.9, reasoning_steps=[]
            )

        assert "cannot be empty" in str(exc_info.value)

    def test_whitespace_cleaning(self):
        """Test cleaning of whitespace in reasoning steps."""
        extraction = ReasoningChainExtraction(
            final_answer="42",
            confidence=0.9,
            reasoning_steps=["  Step 1  ", "", "Step 2"],
        )

        assert len(extraction.reasoning_steps) == 2
        assert extraction.reasoning_steps == ["Step 1", "Step 2"]


class TestExtractionModelFactory:
    """Test suite for get_extraction_model factory function."""

    def test_get_base_model(self):
        """Test getting base extraction model."""
        model_class = get_extraction_model("base")
        assert model_class == BaseAnswerExtraction

    def test_get_multiple_choice_model(self):
        """Test getting multiple choice extraction model."""
        model_class = get_extraction_model("multiple_choice")
        assert model_class == MultipleChoiceExtraction

    def test_get_numerical_model(self):
        """Test getting numerical extraction model."""
        model_class = get_extraction_model("numerical")
        assert model_class == NumericalExtraction

    def test_case_insensitive_lookup(self):
        """Test case insensitive model lookup."""
        model_class = get_extraction_model("TEXTUAL")
        assert model_class == TextualExtraction

    def test_unsupported_type_error(self):
        """Test error for unsupported answer type."""
        with pytest.raises(ValueError) as exc_info:
            get_extraction_model("unsupported_type")

        assert "Unsupported answer type" in str(exc_info.value)

    def test_default_model(self):
        """Test default extraction model."""
        assert DefaultExtractionModel == BaseAnswerExtraction
