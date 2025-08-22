"""Pydantic models for structured answer extraction.

This module defines Pydantic models used for parsing and validating
extracted answers from LLM outputs using the Instructor library.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class BaseAnswerExtraction(BaseModel):
    """Base class for all answer extraction models.

    This class provides common fields and validation methods
    that are shared across different answer extraction types.
    """

    final_answer: str = Field(
        description="The extracted final answer from the response"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for the extraction (0.0 to 1.0)"
    )
    extraction_method: str = Field(
        default="instructor",
        description="Method used for extraction (instructor, regex, manual)",
    )

    @field_validator("final_answer")
    @classmethod
    def final_answer_not_empty(cls, v):
        """Validate that final_answer is not empty."""
        if not v or not v.strip():
            raise ValueError("Final answer cannot be empty")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class MultipleChoiceExtraction(BaseAnswerExtraction):
    """Model for extracting multiple choice answers."""

    selected_option: str = Field(
        description="The selected option (e.g., 'A', 'B', 'C', 'D')"
    )
    option_reasoning: str = Field(
        description="Reasoning for why this option was selected"
    )

    @field_validator("selected_option")
    @classmethod
    def validate_option_format(cls, v):
        """Validate that the selected option is in correct format."""
        if not v or not v.strip():
            raise ValueError("Selected option cannot be empty")
        # Convert to uppercase and check if it's a single letter
        option = v.strip().upper()
        if len(option) == 1 and option.isalpha():
            return option
        # Also accept longer options like "Option A" or "A)"
        return v.strip()


class NumericalExtraction(BaseAnswerExtraction):
    """Model for extracting numerical answers."""

    numerical_value: Union[float, int] = Field(
        description="The numerical value extracted from the answer"
    )
    unit: Optional[str] = Field(
        default=None, description="Unit of measurement if applicable"
    )
    calculation_steps: List[str] = Field(
        default_factory=list, description="List of calculation steps if available"
    )

    @field_validator("numerical_value")
    @classmethod
    def validate_numerical_value(cls, v):
        """Validate that numerical value is a valid number."""
        try:
            float(v)
            return v
        except (ValueError, TypeError):
            raise ValueError("Numerical value must be a valid number")


class TextualExtraction(BaseAnswerExtraction):
    """Model for extracting textual answers."""

    answer_type: str = Field(
        description="Type of answer (yes/no, explanation, summary, etc.)"
    )
    key_points: List[str] = Field(
        default_factory=list, description="Key points or main ideas from the answer"
    )

    @field_validator("answer_type")
    @classmethod
    def validate_answer_type(cls, v):
        """Validate answer type is not empty."""
        if not v or not v.strip():
            raise ValueError("Answer type cannot be empty")
        return v.strip().lower()


class YesNoExtraction(BaseAnswerExtraction):
    """Model for extracting yes/no answers."""

    yes_no_answer: bool = Field(
        description="Boolean representation of the yes/no answer"
    )
    reasoning: str = Field(description="Reasoning behind the yes/no answer")


class ListExtraction(BaseAnswerExtraction):
    """Model for extracting list-based answers."""

    items: List[str] = Field(description="List of items extracted from the answer")
    item_count: Optional[int] = Field(
        default=None, description="Number of items in the list"
    )

    @field_validator("items")
    @classmethod
    def validate_items_not_empty(cls, v):
        """Validate that items list is not empty."""
        if not v:
            raise ValueError("Items list cannot be empty")
        # Remove empty strings
        return [item.strip() for item in v if item.strip()]

    @model_validator(mode="after")
    def set_item_count(self):
        """Set item count based on items list."""
        if self.items:
            self.item_count = len(self.items)
        return self


class ReasoningChainExtraction(BaseAnswerExtraction):
    """Model for extracting answers with reasoning chains."""

    reasoning_steps: List[str] = Field(description="Step-by-step reasoning process")
    intermediate_results: List[str] = Field(
        default_factory=list, description="Intermediate results or thoughts"
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list, description="Flags indicating areas of uncertainty"
    )

    @field_validator("reasoning_steps")
    @classmethod
    def validate_reasoning_steps(cls, v):
        """Validate that reasoning steps are provided."""
        if not v:
            raise ValueError("Reasoning steps cannot be empty")
        return [step.strip() for step in v if step.strip()]


# Factory function to get appropriate extraction model
def get_extraction_model(
    answer_type: str, multiple_responses: bool = False
) -> type[BaseAnswerExtraction]:
    """Get the appropriate extraction model based on answer type.

    Args:
        answer_type: Type of answer expected
        multiple_responses: Whether to return a model that handles multiple responses

    Returns:
        Appropriate Pydantic model class

    Raises:
        ValueError: If answer_type is not supported
    """
    model_map = {
        "multiple_choice": MultipleChoiceExtraction,
        "numerical": NumericalExtraction,
        "textual": TextualExtraction,
        "yes_no": YesNoExtraction,
        "list": ListExtraction,
        "reasoning_chain": ReasoningChainExtraction,
        "base": BaseAnswerExtraction,
    }

    if answer_type.lower() not in model_map:
        raise ValueError(f"Unsupported answer type: {answer_type}")

    # Return multiple response wrapper if requested
    if multiple_responses:
        return MultipleAnswerExtraction

    return model_map[answer_type.lower()]


def detect_multiple_tool_calls(response_text: str) -> bool:
    """Detect if response contains multiple tool calls or function calls.

    Args:
        response_text: The LLM response text to analyze

    Returns:
        True if multiple tool calls detected, False otherwise
    """
    # Common patterns that indicate multiple tool calls
    tool_call_patterns = [
        r"function_call.*?function_call",  # Multiple function_call entries
        r"tool_call.*?tool_call",  # Multiple tool_call entries
        r'\{[^}]*"function"[^}]*\}.*?\{[^}]*"function"[^}]*\}',  # JSON function calls
        r"<tool_call>.*?</tool_call>.*?<tool_call>.*?</tool_call>",  # XML tool calls
        r"```[^`]*```.*?```[^`]*```",  # Multiple code blocks
    ]

    import re

    for pattern in tool_call_patterns:
        if re.search(pattern, response_text, re.DOTALL | re.IGNORECASE):
            return True

    # Check for multiple step indicators
    step_indicators = [
        "step 1:",
        "step 2:",
        "step 3:",
        "first,",
        "second,",
        "third,",
        "1.",
        "2.",
        "3.",
        "next,",
        "then,",
        "finally,",
    ]

    step_count = 0
    for indicator in step_indicators:
        if indicator.lower() in response_text.lower():
            step_count += 1
            if step_count >= 2:
                return True

    return False


# Multiple response wrapper models for handling LLM tool calls
class ToolCallResponse(BaseModel):
    """Model for handling individual tool call responses."""

    call_id: Optional[str] = Field(
        default=None, description="Unique identifier for the tool call"
    )
    function_name: Optional[str] = Field(
        default=None, description="Name of the function called"
    )
    content: str = Field(description="Content or result of the tool call")
    success: bool = Field(
        default=True, description="Whether the tool call was successful"
    )


class MultipleAnswerExtraction(BaseModel):
    """Wrapper model for handling multiple answer extractions from tool calls."""

    extractions: List[BaseAnswerExtraction] = Field(
        description="List of individual answer extractions"
    )
    primary_extraction: Optional[BaseAnswerExtraction] = Field(
        default=None, description="Primary extraction result (highest confidence)"
    )
    tool_calls: List[ToolCallResponse] = Field(
        default_factory=list, description="Individual tool call responses"
    )
    combined_confidence: float = Field(
        ge=0.0, le=1.0, description="Combined confidence score across all extractions"
    )

    @model_validator(mode="after")
    def set_primary_extraction(self):
        """Set the primary extraction to the one with highest confidence."""
        if self.extractions:
            # Find extraction with highest confidence
            self.primary_extraction = max(self.extractions, key=lambda x: x.confidence)
            # Calculate combined confidence as weighted average
            if len(self.extractions) > 1:
                total_conf = sum(ext.confidence for ext in self.extractions)
                self.combined_confidence = total_conf / len(self.extractions)
            else:
                self.combined_confidence = self.extractions[0].confidence
        return self


class CombinedExtractionResult(BaseModel):
    """Model for combining multiple extraction results into a single response."""

    final_answer: str = Field(description="Final combined answer from all extractions")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in the combined result"
    )
    extraction_method: str = Field(
        default="instructor_multiple", description="Method used for extraction"
    )
    source_extractions: List[BaseAnswerExtraction] = Field(
        description="Source extractions that were combined"
    )
    reasoning_chain: List[str] = Field(
        default_factory=list,
        description="Combined reasoning steps from all extractions",
    )

    @field_validator("final_answer")
    @classmethod
    def final_answer_not_empty(cls, v):
        """Validate that final_answer is not empty."""
        if not v or not v.strip():
            raise ValueError("Final answer cannot be empty")
        return v.strip()


# Default extraction model for when type is unknown
DefaultExtractionModel = BaseAnswerExtraction
