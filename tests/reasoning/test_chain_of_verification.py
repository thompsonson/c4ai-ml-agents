"""Tests for Chain-of-Verification approach."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.chain_of_verification import ChainOfVerificationReasoning
from ml_agents.utils.api_clients import StandardResponse


@pytest.fixture
def config():
    """Create test configuration."""
    return ExperimentConfig(
        provider="openrouter",
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=512,
    )


@pytest.fixture
def config_multi_step():
    """Create test configuration with multi-step verification enabled."""
    return ExperimentConfig(
        provider="openrouter",
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=512,
        multi_step_verification=True,
        max_reasoning_calls=3,
    )


@pytest.fixture
def mock_client():
    """Create mock API client."""
    client = MagicMock()
    client.generate.return_value = StandardResponse(
        text="Initial Response: The capital of France is Paris.\n"
        "Verification Questions: 1. Is Paris really the capital? 2. Has this changed recently?\n"
        "Verification Analysis: 1. Yes, Paris is the capital. 2. No recent changes.\n"
        "Accuracy Assessment: The initial response is accurate and reliable.\n"
        "Refined Response: Paris is indeed the capital of France.\n"
        "Final Answer: Paris is the capital of France.",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=60,
        completion_tokens=120,
        total_tokens=180,
        generation_time=3.0,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="test-response-cove-123",
        metadata={},
    )
    return client


@pytest.fixture
def mock_client_multi_step():
    """Create mock API client for multi-step verification."""
    client = MagicMock()

    responses = [
        StandardResponse(
            text="Paris is the capital of France.",
            provider="openrouter",
            model="openai/gpt-oss-120b",
            prompt_tokens=20,
            completion_tokens=40,
            total_tokens=60,
            generation_time=1.0,
            parameters={"temperature": 0.3, "max_tokens": 512},
            response_id="initial-response",
            metadata={},
        ),
        StandardResponse(
            text="1. Is Paris currently the capital of France?\n"
            "2. Has there been any recent change in the capital?\n"
            "3. Are there any special administrative considerations?",
            provider="openrouter",
            model="openai/gpt-oss-120b",
            prompt_tokens=30,
            completion_tokens=50,
            total_tokens=80,
            generation_time=1.5,
            parameters={"temperature": 0.3, "max_tokens": 512},
            response_id="verification-questions",
            metadata={},
        ),
        StandardResponse(
            text="1. Yes, Paris is currently the capital of France.\n"
            "2. No, there has been no recent change.\n"
            "3. No special considerations apply.\n"
            "Final Answer: Paris is the capital of France.",
            provider="openrouter",
            model="openai/gpt-oss-120b",
            prompt_tokens=40,
            completion_tokens=60,
            total_tokens=100,
            generation_time=2.0,
            parameters={"temperature": 0.3, "max_tokens": 512},
            response_id="final-verification",
            metadata={},
        ),
    ]

    client.generate.side_effect = responses
    return client


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_init_with_prompt_file(mock_file, mock_create_client, config, mock_client):
    """Test initialization with existing prompt file."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "Test CoVe prompt with {question}"

    reasoning = ChainOfVerificationReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert reasoning.approach_name == "ChainOfVerification"
    assert "Test CoVe prompt" in reasoning.cove_prompt
    assert reasoning.multi_step_verification is False


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", side_effect=FileNotFoundError)
def test_init_with_fallback_prompt(mock_file, mock_create_client, config, mock_client):
    """Test initialization with fallback prompt when file is missing."""
    mock_create_client.return_value = mock_client

    reasoning = ChainOfVerificationReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert "verification" in reasoning.cove_prompt


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_init_multi_step_config(
    mock_file, mock_create_client, config_multi_step, mock_client
):
    """Test initialization with multi-step configuration."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "CoVe prompt: {question}"

    reasoning = ChainOfVerificationReasoning(config_multi_step)

    assert reasoning.multi_step_verification is True
    assert reasoning.max_iterations == 3


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_single_prompt_mode(mock_file, mock_create_client, config, mock_client):
    """Test execution in single-prompt mode."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "CoVe prompt: {question}"

    reasoning = ChainOfVerificationReasoning(config)
    result = reasoning.execute("What is the capital of France?")

    # Verify API client was called once
    mock_client.generate.assert_called_once()
    call_args = mock_client.generate.call_args[0][0]
    assert "What is the capital of France?" in call_args

    # Verify response structure
    assert isinstance(result, StandardResponse)
    assert "Initial Response:" in result.text
    assert result.provider == "openrouter"
    assert result.model == "openai/gpt-oss-120b"

    # Verify metadata enhancement
    assert "reasoning_steps" in result.metadata
    assert "approach_specific_metrics" in result.metadata
    metrics = result.metadata["approach_specific_metrics"]
    assert metrics["verification_mode"] == "single_prompt"
    assert metrics["template_used"] == "chain_of_verification"


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_multi_step_mode(
    mock_file, mock_create_client, config_multi_step, mock_client_multi_step
):
    """Test execution in multi-step mode."""
    mock_create_client.return_value = mock_client_multi_step
    mock_file.return_value.read.return_value = "CoVe prompt: {question}"

    reasoning = ChainOfVerificationReasoning(config_multi_step)
    result = reasoning.execute("What is the capital of France?")

    # Verify API client was called three times
    assert mock_client_multi_step.generate.call_count == 3

    # Verify response structure
    assert isinstance(result, StandardResponse)
    assert "Initial Response:" in result.text
    assert "Verification Questions:" in result.text
    assert "Final Verification:" in result.text

    # Verify token aggregation
    assert result.total_tokens == 240  # 60 + 80 + 100
    assert result.generation_time == 4.5  # 1.0 + 1.5 + 2.0

    # Verify metadata
    metrics = result.metadata["approach_specific_metrics"]
    assert metrics["verification_mode"] == "multi_step"
    assert metrics["verification_steps"] == 3
    assert metrics["total_api_calls"] == 3
    assert "multi_step_details" in result.metadata


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_metadata_analysis(mock_file, mock_create_client, config, mock_client):
    """Test metadata analysis for verification characteristics."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "CoVe prompt: {question}"

    reasoning = ChainOfVerificationReasoning(config)
    result = reasoning.execute("Test question")

    metrics = result.metadata["approach_specific_metrics"]

    # Check that all verification components are detected
    assert metrics["contains_initial_response"] is True
    assert metrics["contains_verification_questions"] is True
    assert metrics["contains_accuracy_assessment"] is True
    assert metrics["contains_refined_response"] is True
    assert metrics["verification_questions_count"] > 0
    assert 0.0 <= metrics["verification_quality_score"] <= 1.0


class TestVerificationAnalysis:
    """Test verification analysis methods."""

    @pytest.fixture
    def reasoning_instance(self, config, mock_client):
        """Create reasoning instance for testing."""
        with patch("ml_agents.utils.api_clients.create_api_client") as mock_create:
            mock_create.return_value = mock_client
            with patch("builtins.open", side_effect=FileNotFoundError):
                return ChainOfVerificationReasoning(config)

    def test_count_verification_questions(self, reasoning_instance):
        """Test counting of verification questions."""
        text_with_questions = """
        1. Is this answer correct?
        2. Are there any contradictions?
        3. What evidence supports this claim?
        4. Could there be alternative explanations?
        """

        count = reasoning_instance._count_verification_questions(text_with_questions)
        assert count >= 4  # Should detect multiple questions

    def test_count_verification_questions_minimal(self, reasoning_instance):
        """Test counting with minimal question content."""
        text_minimal = "This is a statement without questions."

        count = reasoning_instance._count_verification_questions(text_minimal)
        assert count >= 0  # Should handle minimal content gracefully

    def test_analyze_verification_quality(self, reasoning_instance):
        """Test verification quality analysis."""
        high_quality_text = """
        I need to verify this answer carefully and check for accuracy.
        Let me evaluate the evidence and validate each claim systematically.
        After verification, I can refine my response to be more reliable.
        """

        quality = reasoning_instance._analyze_verification_quality(high_quality_text)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.0  # Should detect verification quality indicators

    def test_analyze_verification_quality_low(self, reasoning_instance):
        """Test verification quality with low-quality text."""
        low_quality_text = "This is a simple answer."

        quality = reasoning_instance._analyze_verification_quality(low_quality_text)
        assert 0.0 <= quality <= 1.0

    def test_has_initial_response(self, reasoning_instance):
        """Test initial response detection."""
        text_with_initial = "Initial response: This is my first answer to the question."
        text_without_initial = "This is just a regular response."

        assert reasoning_instance._has_initial_response(text_with_initial) is True
        assert reasoning_instance._has_initial_response(text_without_initial) is False

    def test_has_verification_questions(self, reasoning_instance):
        """Test verification questions detection."""
        text_with_questions = (
            "Verification questions: Is this correct? Are we sure about this?"
        )
        text_without_questions = "This is just a regular response."

        assert (
            reasoning_instance._has_verification_questions(text_with_questions) is True
        )
        assert (
            reasoning_instance._has_verification_questions(text_without_questions)
            is False
        )

    def test_has_accuracy_assessment(self, reasoning_instance):
        """Test accuracy assessment detection."""
        text_with_assessment = "The accuracy of this response is high and reliable."
        text_without_assessment = "This is just a regular response."

        assert reasoning_instance._has_accuracy_assessment(text_with_assessment) is True
        assert (
            reasoning_instance._has_accuracy_assessment(text_without_assessment)
            is False
        )

    def test_has_refined_response(self, reasoning_instance):
        """Test refined response detection."""
        text_with_refined = "After verification, my refined answer is more accurate."
        text_without_refined = "This is just a regular response."

        assert reasoning_instance._has_refined_response(text_with_refined) is True
        assert reasoning_instance._has_refined_response(text_without_refined) is False


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_error_handling(mock_file, mock_create_client, config):
    """Test error handling during execution."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = Exception("API Error")
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "CoVe prompt: {question}"

    reasoning = ChainOfVerificationReasoning(config)

    with pytest.raises(Exception, match="API Error"):
        reasoning.execute("Test question")


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_inheritance_from_base(mock_file, mock_create_client, config, mock_client):
    """Test that the class properly inherits from BaseReasoning."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "CoVe prompt: {question}"

    reasoning = ChainOfVerificationReasoning(config)

    # Should have inherited methods from BaseReasoning
    assert hasattr(reasoning, "_enhance_metadata")
    assert hasattr(reasoning, "execute")
    assert reasoning.approach_name == "ChainOfVerification"


@patch("ml_agents.gents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_complex_verification_response(
    mock_file, mock_create_client, config, mock_client
):
    """Test with a complex verification response."""
    complex_response = StandardResponse(
        text="""Initial Response: Climate change is primarily caused by human activities.

        Verification Questions:
        1. What scientific evidence supports this claim?
        2. Are there alternative explanations for climate change?
        3. How strong is the scientific consensus on this topic?
        4. What are the main human activities contributing to this?
        5. How reliable are the measurement methods used?

        Verification Analysis:
        1. Multiple peer-reviewed studies show correlation between CO2 and temperature
        2. Natural causes alone cannot explain current warming rate
        3. 97% of climate scientists agree on human causation
        4. Fossil fuel burning, deforestation, and industrial processes
        5. Multiple independent measurement systems confirm the data

        Accuracy Assessment: The initial response is accurate and well-supported by evidence.

        Refined Response: Climate change is primarily caused by human activities,
        specifically greenhouse gas emissions from fossil fuel combustion, with
        strong scientific consensus supporting this conclusion.

        Final Answer: Human activities are the primary cause of current climate change.""",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=100,
        completion_tokens=250,
        total_tokens=350,
        generation_time=4.0,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="complex-response-789",
        metadata={},
    )

    mock_client.generate.return_value = complex_response
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "CoVe prompt: {question}"

    reasoning = ChainOfVerificationReasoning(config)
    result = reasoning.execute("What causes climate change?")

    # Verify complex analysis
    metrics = result.metadata["approach_specific_metrics"]
    assert metrics["verification_questions_count"] >= 5
    assert metrics["verification_quality_score"] > 0.5  # Should be high quality
    assert metrics["contains_initial_response"] is True
    assert metrics["contains_verification_questions"] is True
    assert metrics["contains_accuracy_assessment"] is True
    assert metrics["contains_refined_response"] is True
