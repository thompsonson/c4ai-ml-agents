# Test Specification: Simplified Instructor Integration Test Suite

## Purpose

Verify that the simplified InstructorClientManager correctly selects appropriate modes (TOOLS or JSON) based on provider capabilities and implements basic TOOLS → JSON fallback across all ML Agents supported providers.

## Overview

The simplified implementation uses:

- **Primary modes**: ANTHROPIC_TOOLS, TOOLS, or JSON based on provider
- **Simple fallback**: TOOLS → JSON when primary mode fails
- **Provider detection**: Uses existing `self.client.provider`
- **Unified extraction**: All reasoning types use same base extraction model

## Test Categories

### 1. Provider Mode Selection Tests

**Test: `test_provider_mode_selection`**

```python
@pytest.mark.parametrize("provider,expected_primary,expected_fallback", [
    ("anthropic", "ANTHROPIC_TOOLS", "ANTHROPIC_JSON"),
    ("openrouter", "TOOLS", "JSON"),
    ("cohere", "JSON", "JSON"),
    ("local-openai", "TOOLS", "JSON"),
])
def test_provider_mode_selection(provider, expected_primary, expected_fallback):
    """Verify correct mode selection based on provider."""
    manager = InstructorClientManager(mock_api_client(), provider)
    assert manager.get_primary_mode() == expected_primary
    assert manager.get_fallback_mode() == expected_fallback
```

**Test: `test_instructor_client_creation`**

```python
@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "cohere", "local-openai"])
def test_instructor_client_creation(provider):
    """Verify Instructor client creation for each provider."""
    manager = InstructorClientManager(mock_api_client(), provider)
    client = manager.get_instructor_client()

    # Verify client is properly patched
    assert hasattr(client, 'chat')
    assert hasattr(client.chat, 'completions')
    assert hasattr(client.chat.completions, 'create')
```

### 2. Fallback Mechanism Tests

**Test: `test_simple_fallback_mechanism`**

```python
@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "local-openai"])
def test_simple_fallback_mechanism(provider):
    """Verify TOOLS → JSON fallback works."""
    manager = InstructorClientManager(mock_api_client(), provider)

    # Mock primary mode failure
    with patch.object(manager, 'get_instructor_client') as mock_client:
        mock_primary = Mock()
        mock_primary.chat.completions.create.side_effect = Exception("TOOLS mode failed")
        mock_fallback = Mock()
        mock_fallback.chat.completions.create.return_value = Mock()

        mock_client.side_effect = [mock_primary, mock_fallback]

        # Should succeed with fallback
        result = manager.extract_structured_response(
            messages=[{"role": "user", "content": "test"}],
            response_model=ChainOfThoughtExtraction
        )

        # Verify both modes attempted
        assert mock_client.call_count == 2
        assert result is not None
```

**Test: `test_cohere_no_fallback`**

```python
def test_cohere_no_fallback():
    """Verify Cohere doesn't attempt fallback (primary == fallback)."""
    manager = InstructorClientManager(mock_api_client(), "cohere")

    with patch.object(manager, 'get_instructor_client') as mock_client:
        mock_client.return_value.chat.completions.create.side_effect = Exception("JSON failed")

        # Should raise exception (no fallback available)
        with pytest.raises(Exception, match="JSON failed"):
            manager.extract_structured_response(
                messages=[{"role": "user", "content": "test"}],
                response_model=ChainOfThoughtExtraction
            )
```

### 3. Integration with BaseReasoning Tests

**Test: `test_base_reasoning_instructor_integration`**

```python
@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "cohere", "local-openai"])
def test_base_reasoning_instructor_integration(provider):
    """Test BaseReasoning uses InstructorClientManager correctly."""
    client = mock_api_client()
    client.provider = provider

    reasoning = ChainOfThoughtReasoning(client)

    # Verify InstructorClientManager is initialized
    assert hasattr(reasoning, 'instructor_manager')
    assert reasoning.instructor_manager.provider == provider

    # Mock extraction
    with patch.object(reasoning.instructor_manager, 'extract_structured_response') as mock_extract:
        mock_extract.return_value = Mock(
            full_reasoning_text="Step 1: Think...",
            answer_value="42",
            confidence=0.9,
            reasoning_type="chain_of_thought"
        )

        response = reasoning.execute("What is 6 * 7?")

        # Verify structured extraction was used
        mock_extract.assert_called_once()
        assert response.extracted_answer == "42"
        assert response.metadata["reasoning_type"] == "chain_of_thought"
```

### 4. Extraction Model Tests

**Test: `test_answer_value_cleaning`**

```python
@pytest.mark.parametrize("input_answer,expected_clean", [
    ("The answer is 4", "4"),
    ("Final answer: yes", "yes"),
    ("Therefore, 42.5", "42.5"),
    ("Answer: A", "A"),
    ("So, true", "true"),
    ("Hence, Paris", "Paris"),
    ("4", "4"),  # Already clean
])
def test_answer_value_cleaning(input_answer, expected_clean):
    """Test answer cleaning removes common prefixes."""
    extraction = ChainOfThoughtExtraction(
        full_reasoning_text="Some reasoning...",
        answer_value=input_answer,
        confidence=0.8
    )
    assert extraction.answer_value == expected_clean
```

**Test: `test_extraction_model_validation`**

```python
def test_extraction_model_validation():
    """Test Pydantic model validation works correctly."""
    # Valid extraction
    valid_extraction = ChainOfThoughtExtraction(
        full_reasoning_text="Step by step reasoning",
        answer_value="42",
        confidence=0.9
    )
    assert valid_extraction.reasoning_type == "chain_of_thought"

    # Invalid confidence
    with pytest.raises(ValidationError):
        ChainOfThoughtExtraction(
            full_reasoning_text="reasoning",
            answer_value="42",
            confidence=1.5  # > 1.0
        )

    # Empty answer
    with pytest.raises(ValidationError):
        ChainOfThoughtExtraction(
            full_reasoning_text="reasoning",
            answer_value="",  # Empty
            confidence=0.8
        )
```

### 5. Cross-Reasoning Type Tests

**Test: `test_all_reasoning_types_work`**

```python
@pytest.mark.parametrize("reasoning_type,extraction_model", [
    ("none", NoneReasoningExtraction),
    ("chainofthought", ChainOfThoughtExtraction),
    ("treeofthought", TreeOfThoughtExtraction),
    ("programofthought", ProgramOfThoughtExtraction),
    ("reflection", ReflectionExtraction),
    ("chainofverification", ChainOfVerificationExtraction),
])
def test_all_reasoning_types_work(reasoning_type, extraction_model):
    """Verify all reasoning types have correct extraction models."""
    # Test model mapping
    assert REASONING_EXTRACTION_MODELS[reasoning_type] == extraction_model

    # Test model creation
    model_instance = extraction_model(
        full_reasoning_text="Test reasoning",
        answer_value="test",
        confidence=0.8
    )
    assert model_instance.reasoning_type == reasoning_type
```

### 6. Provider-Specific Behavior Tests

**Test: `test_anthropic_tools_mode`**

```python
def test_anthropic_tools_mode():
    """Test Anthropic uses TOOLS mode correctly."""
    manager = InstructorClientManager(mock_anthropic_client(), "anthropic")

    with patch('instructor.from_anthropic') as mock_instructor:
        mock_instructor.return_value = Mock()
        client = manager.get_instructor_client("ANTHROPIC_TOOLS")

        # Verify correct mode passed
        mock_instructor.assert_called_with(
            manager.api_client,
            mode=instructor.Mode.ANTHROPIC_TOOLS
        )
```

**Test: `test_openrouter_compatibility`**

```python
def test_openrouter_compatibility():
    """Test OpenRouter uses OpenAI-compatible client."""
    manager = InstructorClientManager(mock_openrouter_client(), "openrouter")

    with patch('instructor.from_openai') as mock_instructor:
        mock_instructor.return_value = Mock()
        client = manager.get_instructor_client("TOOLS")

        # Verify uses from_openai for OpenRouter
        mock_instructor.assert_called_with(
            manager.api_client,
            mode=instructor.Mode.TOOLS
        )
```

### 7. Error Handling Tests

**Test: `test_unsupported_provider_error`**

```python
def test_unsupported_provider_error():
    """Test error handling for unsupported providers."""
    with pytest.raises(ValueError, match="Unsupported provider: unknown"):
        InstructorClientManager(mock_api_client(), "unknown")
```

**Test: `test_application_level_fallback`**

```python
def test_application_level_fallback():
    """Test reasoning classes fallback to original client.generate() if Instructor fails."""
    client = mock_api_client()
    client.provider = "anthropic"
    client.generate.return_value = Mock(text="Fallback response: answer is 42")

    reasoning = ChainOfThoughtReasoning(client)

    # Mock complete Instructor failure
    with patch.object(reasoning.instructor_manager, 'extract_structured_response') as mock_extract:
        mock_extract.side_effect = Exception("Complete Instructor failure")

        response = reasoning.execute("What is 6 * 7?")

        # Should use fallback
        client.generate.assert_called_once()
        assert "Fallback response" in response.text
```

## Test Implementation Framework

```python
import pytest
from unittest.mock import Mock, patch
from ml_agents.utils.instructor_clients import InstructorClientManager
from ml_agents.utils.reasoning_extraction import (
    ChainOfThoughtExtraction,
    NoneReasoningExtraction,
    TreeOfThoughtExtraction,
    REASONING_EXTRACTION_MODELS
)
from ml_agents.reasoning.chain_of_thought import ChainOfThoughtReasoning

class TestSimplifiedInstructorIntegration:
    """Test suite for simplified Instructor implementation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_providers = ["anthropic", "openrouter", "cohere", "local-openai"]

    def mock_api_client(self, provider="anthropic"):
        """Create mock API client."""
        client = Mock()
        client.provider = provider
        client.model = f"test-model-{provider}"
        client.temperature = 0.1
        client.max_tokens = 500
        client.generate.return_value = Mock(text="Fallback response")
        return client

    def mock_anthropic_client(self):
        """Create mock Anthropic client."""
        return self.mock_api_client("anthropic")

    def mock_openrouter_client(self):
        """Create mock OpenRouter client."""
        return self.mock_api_client("openrouter")

    @pytest.mark.parametrize("provider,reasoning_type", [
        (provider, reasoning)
        for provider in ["anthropic", "openrouter", "cohere", "local-openai"]
        for reasoning in ["none", "chainofthought", "treeofthought"]
    ])
    def test_provider_reasoning_combinations(self, provider, reasoning_type):
        """Test all provider-reasoning combinations work."""
        client = self.mock_api_client(provider)
        manager = InstructorClientManager(client, provider)
        extraction_model = REASONING_EXTRACTION_MODELS[reasoning_type]

        # Mock successful extraction
        with patch.object(manager, 'get_instructor_client') as mock_get_client:
            mock_instructor = Mock()
            mock_instructor.chat.completions.create.return_value = extraction_model(
                full_reasoning_text="Test reasoning",
                answer_value="test_answer",
                confidence=0.8
            )
            mock_get_client.return_value = mock_instructor

            result = manager.extract_structured_response(
                messages=[{"role": "user", "content": "test"}],
                response_model=extraction_model
            )

            assert isinstance(result, extraction_model)
            assert result.reasoning_type == reasoning_type
```

## Success Criteria

1. **Correct Provider Mode Selection**: Each provider uses appropriate primary/fallback modes
2. **Simple Fallback Works**: TOOLS → JSON fallback succeeds when primary fails
3. **Clean Answer Extraction**: Answer values are properly cleaned of prefixes
4. **Metadata Tracking**: Response includes reasoning_type, confidence, instructor_mode
5. **Cross-Provider Consistency**: Same extraction quality across all providers
6. **Graceful Error Handling**: Application-level fallback prevents failures

## Coverage Requirements

- All 4 supported providers
- All 6+ reasoning types
- Primary and fallback modes
- Answer cleaning edge cases
- Error scenarios and fallbacks

This simplified test suite validates the core functionality without the complexity of the original provider-aware architecture while ensuring robust extraction across all supported providers and reasoning approaches.
