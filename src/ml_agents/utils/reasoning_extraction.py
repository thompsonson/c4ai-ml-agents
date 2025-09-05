"""Pydantic models for structured reasoning and answer extraction using Instructor.

This module defines reasoning-specific Pydantic models used for extracting
structured data from LLM responses using the Instructor library. Each reasoning
approach has its own specialized model that separates the full reasoning process
from the final answer value.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal


class ReasoningExtraction(BaseModel):
    """Universal base model for extracting reasoning and answers.

    This base class provides the common structure for all reasoning approaches,
    ensuring consistent separation of reasoning process from answer values.
    """

    full_reasoning_text: str = Field(
        description="The COMPLETE reasoning process including ALL steps, thoughts, "
        "calculations, and explanations. Include everything EXCEPT the final answer value."
    )

    answer_value: str = Field(
        description="ONLY the actual answer value. Examples: '4', 'yes', 'A', 'true', '42.5'. "
        "Do NOT include explanatory text like 'The answer is' or 'Therefore,'."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the answer_value is correctly extracted (0.0 to 1.0)",
    )

    extraction_method: str = Field(
        default="instructor", description="Method used for extraction"
    )

    @field_validator("answer_value")
    @classmethod
    def clean_answer_value(cls, v: str) -> str:
        """Remove common prefixes and clean the answer."""
        if not v:
            raise ValueError("Answer value cannot be empty")

        # Clean common prefixes that might slip through
        prefixes_to_remove = [
            "the answer is ",
            "answer: ",
            "final answer: ",
            "therefore, ",
            "thus, ",
            "so, ",
            "hence, ",
            "the final answer to the question",
        ]

        clean = v.strip().lower()
        for prefix in prefixes_to_remove:
            if clean.startswith(prefix):
                v = v[len(prefix) :].strip()
                clean = v.lower()

        # Remove trailing periods from single word answers
        if len(v.split()) == 1 and v.endswith("."):
            v = v[:-1]

        return v.strip()

    @field_validator("full_reasoning_text")
    @classmethod
    def validate_reasoning_text(cls, v: str) -> str:
        """Ensure reasoning text is not empty."""
        if not v or not v.strip():
            raise ValueError("Reasoning text cannot be empty")
        return v.strip()


class NoneReasoningExtraction(ReasoningExtraction):
    """Extraction model for None (baseline) reasoning.

    This model handles the baseline case where no specific reasoning
    methodology is applied.
    """

    reasoning_type: Literal["none"] = "none"


class ChainOfThoughtExtraction(ReasoningExtraction):
    """Extraction model for Chain of Thought reasoning.

    This model captures the specific characteristics of Chain of Thought
    reasoning, including step counting and structure analysis.
    """

    reasoning_type: Literal["chain_of_thought"] = "chain_of_thought"

    step_count: int = Field(
        ge=0, description="Number of reasoning steps identified in the response"
    )

    contains_numbered_steps: bool = Field(
        description="Whether the reasoning contains numbered steps (1., 2., etc.)"
    )

    @field_validator("step_count")
    @classmethod
    def validate_step_count(cls, v: int) -> int:
        """Ensure step count is reasonable."""
        if v < 0:
            raise ValueError("Step count cannot be negative")
        # Cap at reasonable maximum
        return min(v, 50)


class TreeOfThoughtExtraction(ReasoningExtraction):
    """Extraction model for Tree of Thought reasoning.

    This model captures the branching nature of Tree of Thought reasoning.
    """

    reasoning_type: Literal["tree_of_thought"] = "tree_of_thought"

    branches_explored: int = Field(
        ge=1, description="Number of reasoning branches explored"
    )

    selected_branch: str = Field(
        description="Which reasoning branch was selected for the final answer"
    )


class ProgramOfThoughtExtraction(ReasoningExtraction):
    """Extraction model for Program of Thought reasoning.

    This model handles reasoning that includes code or computational elements.
    """

    reasoning_type: Literal["program_of_thought"] = "program_of_thought"

    contains_code: bool = Field(
        description="Whether the reasoning contains code or pseudocode"
    )

    code_blocks: List[str] = Field(
        default_factory=list, description="List of code blocks found in the reasoning"
    )


class ReflectionExtraction(ReasoningExtraction):
    """Extraction model for Reflection reasoning.

    This model captures the self-evaluation aspects of reflection reasoning.
    """

    reasoning_type: Literal["reflection"] = "reflection"

    reflection_iterations: int = Field(
        ge=1, description="Number of reflection iterations performed"
    )

    self_corrections: List[str] = Field(
        default_factory=list,
        description="List of self-corrections made during reflection",
    )


class ChainOfVerificationExtraction(ReasoningExtraction):
    """Extraction model for Chain of Verification reasoning.

    This model captures the verification steps in CoVe reasoning.
    """

    reasoning_type: Literal["chain_of_verification"] = "chain_of_verification"

    verification_steps: int = Field(
        ge=0, description="Number of verification steps performed"
    )

    verification_results: List[str] = Field(
        default_factory=list, description="Results of each verification step"
    )


# Mapping of reasoning approaches to extraction models
REASONING_EXTRACTION_MODELS = {
    "none": NoneReasoningExtraction,
    "chainofthought": ChainOfThoughtExtraction,
    "treeofthought": TreeOfThoughtExtraction,
    "programofthought": ProgramOfThoughtExtraction,
    "reflection": ReflectionExtraction,
    "chainofverification": ChainOfVerificationExtraction,
}


def get_reasoning_extraction_model(reasoning_type: str) -> type[ReasoningExtraction]:
    """Get the appropriate extraction model for a reasoning type.

    Args:
        reasoning_type: The type of reasoning approach

    Returns:
        The appropriate Pydantic model class

    Raises:
        ValueError: If reasoning_type is not supported
    """
    model = REASONING_EXTRACTION_MODELS.get(reasoning_type.lower())
    if not model:
        raise ValueError(f"Unsupported reasoning type: {reasoning_type}")
    return model


def create_reasoning_prompt_suffix(reasoning_type: str) -> str:
    """Create reasoning-type-specific prompt instructions.

    Args:
        reasoning_type: The type of reasoning approach

    Returns:
        Additional prompt instructions specific to the reasoning type
    """
    suffix_map = {
        "none": """
Provide your complete response and the final answer value separately.
""",
        "chainofthought": """
Think through this step by step. Count your reasoning steps and note if you use numbered steps.
Provide your complete reasoning and the final answer value separately.
""",
        "treeofthought": """
Explore multiple reasoning paths. Indicate how many branches you explored and which one you selected.
Provide your complete reasoning and the final answer value separately.
""",
        "programofthought": """
Use computational thinking. Include any code or calculations in your reasoning.
Provide your complete reasoning and the final answer value separately.
""",
        "reflection": """
Reflect on your reasoning process. Make corrections if needed and count your reflection iterations.
Provide your complete reasoning and the final answer value separately.
""",
        "chainofverification": """
Verify your reasoning with additional steps. Count your verification steps.
Provide your complete reasoning and the final answer value separately.
""",
    }

    return suffix_map.get(reasoning_type.lower(), suffix_map["none"])
