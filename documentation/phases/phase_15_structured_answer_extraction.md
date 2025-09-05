# Phase 15: Simplified Structured Answer Extraction with Instructor

## Overview

This phase implements a streamlined answer extraction system using Pydantic models and the Instructor library to cleanly separate reasoning processes from final answers across all reasoning approaches.

## Problem Statement

The current answer extraction system extracts entire sentences as answers (e.g., "The final answer to the question 'What is 2 + 2?' is 4." instead of just "4"). Previous RegEx-based approaches were unreliable and fragile.

## Solution Design

### Core Principles

1. **Structured Output**: Use Instructor with Pydantic models for reliable extraction
2. **Simple Provider Support**: Support all providers with basic TOOLS → JSON fallback
3. **Clean Answer Separation**: Extract only answer values, preserve full reasoning
4. **Minimal Complexity**: Keep the implementation simple and maintainable

### Provider Support Matrix

| Provider | Primary Mode | Fallback Mode |
|----------|--------------|---------------|
| Anthropic | ANTHROPIC_TOOLS | ANTHROPIC_JSON |
| OpenRouter | TOOLS | JSON |
| Cohere | JSON | - |
| Local-OpenAI | TOOLS | JSON |

## Implementation Details

### 1. Extraction Models

```python
# New file: src/ml_agents/utils/reasoning_extraction.py
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal

class ReasoningExtraction(BaseModel):
    """Base model for extracting reasoning and answers."""

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
        description="Confidence that the answer_value is correctly extracted (0.0 to 1.0)"
    )

    @field_validator("answer_value")
    @classmethod
    def clean_answer_value(cls, v: str) -> str:
        """Remove common prefixes and clean the answer."""
        if not v:
            raise ValueError("Answer value cannot be empty")

        # Clean common prefixes
        prefixes_to_remove = [
            "the answer is",
            "answer:",
            "final answer:",
            "therefore,",
            "thus,",
            "so,",
            "hence,"
        ]

        clean = v.strip().lower()
        for prefix in prefixes_to_remove:
            if clean.startswith(prefix):
                v = v[len(prefix):].strip()
                clean = v.lower()

        return v.strip()

class NoneReasoningExtraction(ReasoningExtraction):
    """Extraction model for None (baseline) reasoning."""
    reasoning_type: Literal["none"] = "none"

class ChainOfThoughtExtraction(ReasoningExtraction):
    """Extraction model for Chain of Thought reasoning."""
    reasoning_type: Literal["chain_of_thought"] = "chain_of_thought"

class TreeOfThoughtExtraction(ReasoningExtraction):
    """Extraction model for Tree of Thought reasoning."""
    reasoning_type: Literal["tree_of_thought"] = "tree_of_thought"

class ProgramOfThoughtExtraction(ReasoningExtraction):
    """Extraction model for Program of Thought reasoning."""
    reasoning_type: Literal["program_of_thought"] = "program_of_thought"

class ReflectionExtraction(ReasoningExtraction):
    """Extraction model for Reflection reasoning."""
    reasoning_type: Literal["reflection"] = "reflection"

class ChainOfVerificationExtraction(ReasoningExtraction):
    """Extraction model for Chain of Verification reasoning."""
    reasoning_type: Literal["chain_of_verification"] = "chain_of_verification"

# Mapping of reasoning approaches to extraction models
REASONING_EXTRACTION_MODELS = {
    "none": NoneReasoningExtraction,
    "chainofthought": ChainOfThoughtExtraction,
    "treeofthought": TreeOfThoughtExtraction,
    "programofthought": ProgramOfThoughtExtraction,
    "reflection": ReflectionExtraction,
    "chainofverification": ChainOfVerificationExtraction,
}
```

### 2. Instructor Client Manager

```python
# New file: src/ml_agents/utils/instructor_clients.py
import instructor
from typing import Type, Any
from pydantic import BaseModel

class InstructorClientManager:
    """Manages Instructor client configuration with simple fallback."""

    def __init__(self, api_client, provider: str):
        self.api_client = api_client
        self.provider = provider

    def get_instructor_client(self, mode: str = None):
        """Get Instructor client for provider with specified mode."""
        if mode is None:
            mode = self.get_primary_mode()

        if self.provider == "anthropic":
            return instructor.from_anthropic(self.api_client, mode=getattr(instructor.Mode, mode))
        elif self.provider == "openrouter":
            return instructor.from_openai(self.api_client, mode=getattr(instructor.Mode, mode))
        elif self.provider == "cohere":
            return instructor.from_cohere(self.api_client, mode=getattr(instructor.Mode, mode))
        elif self.provider == "local-openai":
            return instructor.from_openai(self.api_client, mode=getattr(instructor.Mode, mode))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_primary_mode(self) -> str:
        """Get primary mode for provider."""
        mode_map = {
            "anthropic": "ANTHROPIC_TOOLS",
            "openrouter": "TOOLS",
            "cohere": "JSON",
            "local-openai": "TOOLS",
        }
        return mode_map[self.provider]

    def get_fallback_mode(self) -> str:
        """Get fallback mode for provider."""
        fallback_map = {
            "anthropic": "ANTHROPIC_JSON",
            "openrouter": "JSON",
            "cohere": "JSON",  # Same as primary
            "local-openai": "JSON",
        }
        return fallback_map[self.provider]

    def extract_structured_response(self,
                                  messages: list,
                                  response_model: Type[BaseModel],
                                  **kwargs) -> Any:
        """Extract structured response with simple fallback."""
        try:
            # Try primary mode
            client = self.get_instructor_client(self.get_primary_mode())
            return client.chat.completions.create(
                model=self.api_client.model,
                response_model=response_model,
                messages=messages,
                **kwargs
            )
        except Exception:
            # Simple fallback to secondary mode
            if self.get_primary_mode() != self.get_fallback_mode():
                client = self.get_instructor_client(self.get_fallback_mode())
                return client.chat.completions.create(
                    model=self.api_client.model,
                    response_model=response_model,
                    messages=messages,
                    **kwargs
                )
            else:
                raise  # Re-raise if no fallback available
```

### 3. Integration with Base Reasoning Class

```python
# Updated base reasoning class
from ml_agents.utils.instructor_clients import InstructorClientManager
from ml_agents.utils.reasoning_extraction import REASONING_EXTRACTION_MODELS

class BaseReasoning:
    def __init__(self, client, approach_name, provider: str):
        self.client = client
        self.approach_name = approach_name
        self.instructor_manager = InstructorClientManager(client, provider)

    def execute(self, prompt: str) -> StandardResponse:
        """Execute reasoning with structured extraction."""
        # Apply reasoning-specific prompt template
        enhanced_prompt = self.apply_prompt_template(prompt)

        # Get reasoning-specific extraction model
        extraction_model = REASONING_EXTRACTION_MODELS[self.approach_name.lower()]

        # Use Instructor for structured response generation
        extraction = self.instructor_manager.extract_structured_response(
            messages=[{"role": "user", "content": enhanced_prompt}],
            response_model=extraction_model,
            temperature=self.client.temperature,
            max_tokens=self.client.max_tokens,
        )

        # Create StandardResponse with structured data
        return StandardResponse(
            text=extraction.full_reasoning_text,
            extracted_answer=extraction.answer_value,
            provider=self.client.provider,
            model=self.client.model,
            total_tokens=0,  # To be filled by API response tracking
            generation_time=0.0,  # To be measured
            parameters={"structured_extraction": True},
            metadata={
                "reasoning_type": extraction.reasoning_type,
                "confidence": extraction.confidence,
                "instructor_mode": self.instructor_manager.get_primary_mode(),
            }
        )
```

### 4. Updated Prompt Templates

Prompts are simplified to work with structured output:

```python
# In chain_of_thought.txt
"""
You are an AI assistant that excels at step-by-step reasoning.

Please analyze the following question carefully and think through it step by step.

Question: {question}

Think through this systematically and provide your reasoning and final answer.
"""
```

## Testing Strategy

### Unit Tests

1. **Extraction Model Tests**
   - Test answer cleaning logic across all models
   - Verify field validation works correctly
   - Test with various answer formats

2. **Provider Mode Tests**
   - Test correct mode selection per provider
   - Verify fallback mechanism works
   - Test with all supported providers

3. **Integration Tests**
   - Test extraction across all reasoning approaches
   - Verify answer separation accuracy
   - Test reasoning preservation

### Test Implementation

```python
import pytest
from ml_agents.utils.instructor_clients import InstructorClientManager
from ml_agents.utils.reasoning_extraction import ChainOfThoughtExtraction

@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "cohere", "local-openai"])
def test_provider_mode_selection(provider):
    """Test correct mode selection for each provider."""
    manager = InstructorClientManager(mock_client(), provider)
    primary_mode = manager.get_primary_mode()

    expected_modes = {
        "anthropic": "ANTHROPIC_TOOLS",
        "openrouter": "TOOLS",
        "cohere": "JSON",
        "local-openai": "TOOLS"
    }

    assert primary_mode == expected_modes[provider]

def test_answer_cleaning():
    """Test answer value cleaning works correctly."""
    test_cases = [
        ("The answer is 4", "4"),
        ("Therefore, 42.5", "42.5"),
        ("Final answer: yes", "yes"),
        ("A", "A"),
    ]

    for input_val, expected in test_cases:
        extraction = ChainOfThoughtExtraction(
            full_reasoning_text="Some reasoning",
            answer_value=input_val,
            confidence=0.9
        )
        assert extraction.answer_value == expected

def test_extraction_with_fallback():
    """Test fallback mechanism works."""
    manager = InstructorClientManager(mock_client(), "local-openai")

    # Mock primary mode failure
    with patch.object(manager, 'get_instructor_client') as mock_client:
        mock_client.side_effect = [Exception("TOOLS not supported"), Mock()]

        # Should succeed with fallback
        result = manager.extract_structured_response(
            messages=[{"role": "user", "content": "test"}],
            response_model=ChainOfThoughtExtraction
        )

        # Verify fallback was attempted
        assert mock_client.call_count == 2
```

## Configuration

```python
class ParsingConfig:
    # Core extraction settings
    use_structured_extraction: bool = True
    structured_extraction_temperature: float = 0.1
    structured_extraction_max_tokens: int = 500

    # Provider settings - passed to reasoning classes
    provider_map: Dict[str, str] = {
        "anthropic": "anthropic",
        "openrouter": "openrouter",
        "cohere": "cohere",
        "local-openai": "local-openai"
    }

    # Quality settings
    confidence_threshold: float = 0.7
    enable_fallback: bool = True
```

## Migration Plan

### Phase 1: Implementation

1. Create extraction models and client manager
2. Update base reasoning class
3. Test with one reasoning approach

### Phase 2: Integration

1. Update all reasoning classes
2. Add provider parameter to reasoning initialization
3. Update configuration

### Phase 3: Testing & Validation

1. Run comprehensive test suite
2. Validate with real experiment data
3. Performance testing

### Phase 4: Deployment

1. Update configuration to enable new extraction
2. Monitor extraction accuracy
3. Full rollout after validation

## Benefits

1. **Clean Answer Extraction**: Only answer values extracted, no explanatory text
2. **Reliable Structure**: Pydantic validation ensures data quality
3. **Simple Architecture**: Minimal complexity, easy to maintain
4. **Provider Support**: Works across all supported providers
5. **Robust Fallback**: Simple TOOLS → JSON fallback for reliability

## Conclusion

This simplified approach achieves the core goal of clean answer extraction using Instructor's structured output capabilities while maintaining a simple, maintainable architecture. The basic fallback mechanism provides reliability without complex provider-specific logic.
