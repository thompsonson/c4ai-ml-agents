"""Tests for Reasoning-as-Planning approach."""

import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.reasoning_as_planning import ReasoningAsPlanningReasoning
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
def mock_client():
    """Create mock API client."""
    client = MagicMock()
    client.generate.return_value = StandardResponse(
        text="Goal Analysis: The objective is to solve this problem systematically.\n"
        "Planning Phase: I will break this into manageable sub-goals.\n"
        "Action Sequence: Step 1: Analyze requirements. Step 2: Design solution.\n"
        "Risk Assessment: Potential challenges include complexity and constraints.\n"
        "Execution Strategy: Implement incrementally with continuous validation.\n"
        "Final Answer: The solution is a structured approach.",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=50,
        completion_tokens=100,
        total_tokens=150,
        generation_time=2.5,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="test-response-123",
        metadata={},
    )
    return client


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_init_with_prompt_file(mock_file, mock_create_client, config, mock_client):
    """Test initialization with existing prompt file."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "Test RAP prompt with {question}"

    reasoning = ReasoningAsPlanningReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert reasoning.approach_name == "ReasoningAsPlanning"
    assert "Test RAP prompt" in reasoning.rap_prompt


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", side_effect=FileNotFoundError)
def test_init_with_fallback_prompt(mock_file, mock_create_client, config, mock_client):
    """Test initialization with fallback prompt when file is missing."""
    mock_create_client.return_value = mock_client

    reasoning = ReasoningAsPlanningReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert "planning problem" in reasoning.rap_prompt


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_basic(mock_file, mock_create_client, config, mock_client):
    """Test basic execution of reasoning approach."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "RAP prompt: {question}"

    reasoning = ReasoningAsPlanningReasoning(config)
    result = reasoning.execute("How do I organize a conference?")

    # Verify API client was called
    mock_client.generate.assert_called_once()
    call_args = mock_client.generate.call_args[0][0]
    assert "How do I organize a conference?" in call_args

    # Verify response structure
    assert isinstance(result, StandardResponse)
    assert result.text.startswith("Goal Analysis:")
    assert result.provider == "openrouter"
    assert result.model == "openai/gpt-3.5-turbo"

    # Verify metadata enhancement
    assert "reasoning_steps" in result.metadata
    assert "approach_specific_metrics" in result.metadata
    assert (
        result.metadata["approach_specific_metrics"]["template_used"]
        == "reasoning_as_planning"
    )


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_metadata_analysis(mock_file, mock_create_client, config, mock_client):
    """Test metadata analysis for planning characteristics."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "RAP prompt: {question}"

    reasoning = ReasoningAsPlanningReasoning(config)
    result = reasoning.execute("Test question")

    metrics = result.metadata["approach_specific_metrics"]

    # Check that all planning components are detected
    assert metrics["contains_goal_analysis"] is True
    assert metrics["contains_action_sequence"] is True
    assert metrics["contains_risk_assessment"] is True
    assert metrics["contains_execution_strategy"] is True
    assert metrics["planning_stages_identified"] > 0
    assert 0.0 <= metrics["planning_quality_score"] <= 1.0


class TestPlanningAnalysis:
    """Test planning analysis methods."""

    @pytest.fixture
    def reasoning_instance(self, config, mock_client):
        """Create reasoning instance for testing."""
        with patch(
            "ml_agents.reasoning.reasoning_as_planning.create_api_client"
        ) as mock_create:
            mock_create.return_value = mock_client
            with patch("builtins.open", side_effect=FileNotFoundError):
                return ReasoningAsPlanningReasoning(config)

    def test_count_planning_stages(self, reasoning_instance):
        """Test counting of planning stages."""
        text_with_stages = """
        Goal Analysis: Define the objective clearly.
        Planning Phase: Break down into steps.
        Action Sequence: 1. First step 2. Second step
        Risk Assessment: Consider potential issues.
        Execution Strategy: Implement systematically.
        """

        stages = reasoning_instance._count_planning_stages(text_with_stages)
        assert stages >= 5  # Should detect multiple planning stages

    def test_count_planning_stages_minimal(self, reasoning_instance):
        """Test counting with minimal planning content."""
        text_minimal = "This is a simple answer without planning structure."

        stages = reasoning_instance._count_planning_stages(text_minimal)
        assert stages >= 0  # Should handle minimal content gracefully

    def test_analyze_planning_quality(self, reasoning_instance):
        """Test planning quality analysis."""
        high_quality_text = """
        This requires a strategic plan to achieve our goals.
        We need to implement a roadmap with clear milestones.
        Execution depends on proper planning and risk mitigation.
        Consider potential obstacles and develop contingency strategies.
        """

        quality = reasoning_instance._analyze_planning_quality(high_quality_text)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.0  # Should detect planning quality indicators

    def test_analyze_planning_quality_low(self, reasoning_instance):
        """Test planning quality with low-quality text."""
        low_quality_text = "This is a simple answer."

        quality = reasoning_instance._analyze_planning_quality(low_quality_text)
        assert 0.0 <= quality <= 1.0

    def test_has_goal_analysis(self, reasoning_instance):
        """Test goal analysis detection."""
        text_with_goals = "The goal is to achieve our objective and define what we need to accomplish."
        text_without_goals = "This is just a regular response."

        assert reasoning_instance._has_goal_analysis(text_with_goals) is True
        assert reasoning_instance._has_goal_analysis(text_without_goals) is False

    def test_has_action_sequence(self, reasoning_instance):
        """Test action sequence detection."""
        text_with_actions = (
            "Step 1: First do this. Step 2: Then do that. Finally complete the task."
        )
        text_without_actions = "This is just a regular response."

        assert reasoning_instance._has_action_sequence(text_with_actions) is True
        assert reasoning_instance._has_action_sequence(text_without_actions) is False

    def test_has_risk_assessment(self, reasoning_instance):
        """Test risk assessment detection."""
        text_with_risks = "Potential risks include challenges and obstacles. Consider mitigation strategies."
        text_without_risks = "This is just a regular response."

        assert reasoning_instance._has_risk_assessment(text_with_risks) is True
        assert reasoning_instance._has_risk_assessment(text_without_risks) is False

    def test_has_execution_strategy(self, reasoning_instance):
        """Test execution strategy detection."""
        text_with_execution = (
            "We need to implement this strategy and execute the plan carefully."
        )
        text_without_execution = "This is just a regular response."

        assert reasoning_instance._has_execution_strategy(text_with_execution) is True
        assert (
            reasoning_instance._has_execution_strategy(text_without_execution) is False
        )


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_error_handling(mock_file, mock_create_client, config):
    """Test error handling during execution."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = Exception("API Error")
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "RAP prompt: {question}"

    reasoning = ReasoningAsPlanningReasoning(config)

    with pytest.raises(Exception, match="API Error"):
        reasoning.execute("Test question")


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_inheritance_from_base(mock_file, mock_create_client, config, mock_client):
    """Test that the class properly inherits from BaseReasoning."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "RAP prompt: {question}"

    reasoning = ReasoningAsPlanningReasoning(config)

    # Should have inherited methods from BaseReasoning
    assert hasattr(reasoning, "_enhance_metadata")
    assert hasattr(reasoning, "execute")
    assert reasoning.approach_name == "ReasoningAsPlanning"


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_complex_planning_response(mock_file, mock_create_client, config, mock_client):
    """Test with a complex planning response."""
    complex_response = StandardResponse(
        text="""Goal Analysis: We need to develop a comprehensive project management strategy.

        Planning Phase: Break down into these sub-goals:
        - Resource allocation
        - Timeline development
        - Risk mitigation

        Action Sequence:
        1. Conduct stakeholder analysis
        2. Define project scope and objectives
        3. Develop work breakdown structure
        4. Create project timeline with milestones
        5. Identify risks and develop contingency plans
        6. Implement monitoring and control processes

        Risk Assessment: Potential challenges include:
        - Resource constraints
        - Timeline pressures
        - Scope creep
        - Technical difficulties

        Execution Strategy: Use agile methodology with iterative development.
        Implement continuous monitoring and adjustment based on feedback.

        Final Answer: A structured project management approach with clear goals,
        systematic planning, risk mitigation, and adaptive execution strategy.""",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=75,
        completion_tokens=200,
        total_tokens=275,
        generation_time=3.2,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="complex-response-456",
        metadata={},
    )

    mock_client.generate.return_value = complex_response
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "RAP prompt: {question}"

    reasoning = ReasoningAsPlanningReasoning(config)
    result = reasoning.execute("How do I manage a complex project?")

    # Verify complex analysis
    metrics = result.metadata["approach_specific_metrics"]
    assert metrics["planning_stages_identified"] >= 5
    assert metrics["planning_quality_score"] > 0.5  # Should be high quality
    assert metrics["contains_goal_analysis"] is True
    assert metrics["contains_action_sequence"] is True
    assert metrics["contains_risk_assessment"] is True
    assert metrics["contains_execution_strategy"] is True
