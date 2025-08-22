"""Tests for Tree-of-Thought approach."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.tree_of_thought import TreeOfThoughtReasoning
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
        text="Problem Decomposition: Breaking down into key components.\n"
        "Reasoning Branches:\n"
        "Approach 1: Direct analytical method\n"
        "Approach 2: Comparative analysis method\n"
        "Approach 3: Systematic evaluation method\n\n"
        "Branch Exploration: Exploring each approach in detail.\n"
        "Path Evaluation: Assessing strengths and weaknesses of each branch.\n"
        "Branch Selection: Choosing the most promising path.\n"
        "Synthesis: Combining insights from the best approaches.\n"
        "Final Answer: Optimal solution based on tree exploration.",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=80,
        completion_tokens=160,
        total_tokens=240,
        generation_time=4.0,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="test-response-tot-123",
        metadata={},
    )
    return client


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_init_with_prompt_file(mock_file, mock_create_client, config, mock_client):
    """Test initialization with existing prompt file."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "Test ToT prompt with {question}"

    reasoning = TreeOfThoughtReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert reasoning.approach_name == "TreeOfThought"
    assert "Test ToT prompt" in reasoning.tot_prompt


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", side_effect=FileNotFoundError)
def test_init_with_fallback_prompt(mock_file, mock_create_client, config, mock_client):
    """Test initialization with fallback prompt when file is missing."""
    mock_create_client.return_value = mock_client

    reasoning = TreeOfThoughtReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert "reasoning paths" in reasoning.tot_prompt


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_basic(mock_file, mock_create_client, config, mock_client):
    """Test basic execution of reasoning approach."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "ToT prompt: {question}"

    reasoning = TreeOfThoughtReasoning(config)
    result = reasoning.execute("What is the best approach to solve this problem?")

    # Verify API client was called
    mock_client.generate.assert_called_once()
    call_args = mock_client.generate.call_args[0][0]
    assert "What is the best approach to solve this problem?" in call_args

    # Verify response structure
    assert isinstance(result, StandardResponse)
    assert "Problem Decomposition:" in result.text
    assert result.provider == "openrouter"
    assert result.model == "openai/gpt-oss-120b"

    # Verify metadata enhancement
    assert "reasoning_steps" in result.metadata
    assert "approach_specific_metrics" in result.metadata
    assert (
        result.metadata["approach_specific_metrics"]["template_used"]
        == "tree_of_thought"
    )


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_metadata_analysis(mock_file, mock_create_client, config, mock_client):
    """Test metadata analysis for tree reasoning characteristics."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "ToT prompt: {question}"

    reasoning = TreeOfThoughtReasoning(config)
    result = reasoning.execute("Test question")

    metrics = result.metadata["approach_specific_metrics"]

    # Check that all tree components are detected
    assert metrics["contains_problem_decomposition"] is True
    assert metrics["contains_reasoning_branches"] is True
    assert metrics["contains_branch_exploration"] is True
    assert metrics["contains_path_evaluation"] is True
    assert metrics["contains_branch_selection"] is True
    assert metrics["contains_synthesis"] is True
    assert metrics["reasoning_branches"] > 0
    assert metrics["branch_depth"] >= 1
    assert metrics["evaluation_instances"] > 0
    assert 0.0 <= metrics["reasoning_quality_score"] <= 1.0


class TestBranchAnalysis:
    """Test branch analysis methods."""

    @pytest.fixture
    def reasoning_instance(self, config, mock_client):
        """Create reasoning instance for testing."""
        with patch("ml_agents.utils.api_clients.create_api_client") as mock_create:
            mock_create.return_value = mock_client
            with patch("builtins.open", side_effect=FileNotFoundError):
                return TreeOfThoughtReasoning(config)

    def test_count_reasoning_branches(self, reasoning_instance):
        """Test counting of reasoning branches."""
        text_with_branches = """
        Approach 1: Direct analysis
        Approach 2: Comparative method
        Alternative 3: Systematic evaluation
        Path 4: Iterative refinement
        """

        branches = reasoning_instance._count_reasoning_branches(text_with_branches)
        assert branches >= 4  # Should detect multiple approaches

    def test_count_reasoning_branches_alternatives(self, reasoning_instance):
        """Test counting with different branch indicators."""
        text_with_alternatives = """
        First approach: Method A
        Second method: Method B
        Another alternative: Method C
        Different way: Method D
        """

        branches = reasoning_instance._count_reasoning_branches(text_with_alternatives)
        assert branches >= 4  # Should detect alternative naming

    def test_count_reasoning_branches_minimal(self, reasoning_instance):
        """Test counting with minimal branch content."""
        text_minimal = "This is a simple answer without clear branches."

        branches = reasoning_instance._count_reasoning_branches(text_minimal)
        assert branches >= 1  # Should return at least 1

    def test_analyze_branch_depth(self, reasoning_instance):
        """Test branch depth analysis."""
        high_depth_text = """
        I will explore each approach in detail and provide thorough analysis.
        This involves comprehensive examination of pros and cons for each method.
        Systematic investigation reveals the strengths and weaknesses.
        """

        depth = reasoning_instance._analyze_branch_depth(high_depth_text)
        assert 1 <= depth <= 5
        assert depth > 2  # Should detect high depth indicators

    def test_analyze_branch_depth_minimal(self, reasoning_instance):
        """Test branch depth with minimal content."""
        low_depth_text = "Simple answer."

        depth = reasoning_instance._analyze_branch_depth(low_depth_text)
        assert depth == 1  # Should return minimal depth

    def test_count_evaluations(self, reasoning_instance):
        """Test counting of evaluation instances."""
        text_with_evaluations = """
        Let me evaluate each approach and assess their strengths.
        Comparing the pros and cons of each method.
        The advantages and disadvantages need careful consideration.
        This approach is better while that one is superior.
        """

        evaluations = reasoning_instance._count_evaluations(text_with_evaluations)
        assert evaluations >= 8  # Should detect multiple evaluation terms

    def test_analyze_reasoning_quality(self, reasoning_instance):
        """Test reasoning quality analysis."""
        high_quality_text = """
        I will explore multiple branches and paths to find the optimal solution.
        Each alternative approach requires detailed investigation and evaluation.
        Comparing the strengths and weaknesses helps select the best option.
        The synthesis of insights leads to the most promising path.
        """

        quality = reasoning_instance._analyze_reasoning_quality(high_quality_text)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.0  # Should detect tree reasoning quality indicators

    def test_analyze_reasoning_quality_low(self, reasoning_instance):
        """Test reasoning quality with low-quality text."""
        low_quality_text = "This is a simple answer."

        quality = reasoning_instance._analyze_reasoning_quality(low_quality_text)
        assert 0.0 <= quality <= 1.0

    def test_has_problem_decomposition(self, reasoning_instance):
        """Test problem decomposition detection."""
        text_with_decomposition = (
            "Breaking down the problem into key components and elements."
        )
        text_without_decomposition = "This is just a regular response."

        assert (
            reasoning_instance._has_problem_decomposition(text_with_decomposition)
            is True
        )
        assert (
            reasoning_instance._has_problem_decomposition(text_without_decomposition)
            is False
        )

    def test_has_reasoning_branches(self, reasoning_instance):
        """Test reasoning branches detection."""
        text_with_branches = (
            "Exploring multiple approaches and different paths to the solution."
        )
        text_without_branches = "This is just a regular response."

        assert reasoning_instance._has_reasoning_branches(text_with_branches) is True
        assert (
            reasoning_instance._has_reasoning_branches(text_without_branches) is False
        )

    def test_has_branch_exploration(self, reasoning_instance):
        """Test branch exploration detection."""
        text_with_exploration = (
            "Exploring each branch in detail and investigating thoroughly."
        )
        text_without_exploration = "This is just a regular response."

        assert reasoning_instance._has_branch_exploration(text_with_exploration) is True
        assert (
            reasoning_instance._has_branch_exploration(text_without_exploration)
            is False
        )

    def test_has_path_evaluation(self, reasoning_instance):
        """Test path evaluation detection."""
        text_with_evaluation = (
            "Evaluating the pros and cons of each approach and comparing strengths."
        )
        text_without_evaluation = "This is just a regular response."

        assert reasoning_instance._has_path_evaluation(text_with_evaluation) is True
        assert reasoning_instance._has_path_evaluation(text_without_evaluation) is False

    def test_has_branch_selection(self, reasoning_instance):
        """Test branch selection detection."""
        text_with_selection = (
            "Selecting the best and most promising approach for the solution."
        )
        text_without_selection = "This is just a regular response."

        assert reasoning_instance._has_branch_selection(text_with_selection) is True
        assert reasoning_instance._has_branch_selection(text_without_selection) is False

    def test_has_synthesis(self, reasoning_instance):
        """Test synthesis detection."""
        text_with_synthesis = (
            "Combining insights and integrating the best elements from each approach."
        )
        text_without_synthesis = "This is just a regular response."

        assert reasoning_instance._has_synthesis(text_with_synthesis) is True
        assert reasoning_instance._has_synthesis(text_without_synthesis) is False


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_error_handling(mock_file, mock_create_client, config):
    """Test error handling during execution."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = Exception("API Error")
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "ToT prompt: {question}"

    reasoning = TreeOfThoughtReasoning(config)

    with pytest.raises(Exception, match="API Error"):
        reasoning.execute("Test question")


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_inheritance_from_base(mock_file, mock_create_client, config, mock_client):
    """Test that the class properly inherits from BaseReasoning."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "ToT prompt: {question}"

    reasoning = TreeOfThoughtReasoning(config)

    # Should have inherited methods from BaseReasoning
    assert hasattr(reasoning, "_enhance_metadata")
    assert hasattr(reasoning, "execute")
    assert reasoning.approach_name == "TreeOfThought"


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_complex_tree_response(mock_file, mock_create_client, config, mock_client):
    """Test with a complex tree-structured response."""
    complex_response = StandardResponse(
        text="""Problem Decomposition:
        Breaking down the challenge into core components: efficiency, accuracy, and scalability.

        Reasoning Branches:

        Branch A: Optimization-focused approach
        - Prioritizes efficiency and speed
        - Uses algorithmic shortcuts
        - May sacrifice some accuracy

        Branch B: Accuracy-first methodology
        - Emphasizes precision and correctness
        - Uses comprehensive validation
        - May be slower but more reliable

        Branch C: Hybrid balanced strategy
        - Combines efficiency and accuracy
        - Uses adaptive algorithms
        - Balances trade-offs dynamically

        Branch D: Scalability-oriented solution
        - Designed for large-scale deployment
        - Uses distributed processing
        - Focuses on horizontal scaling

        Branch Exploration:
        Exploring each approach thoroughly with detailed investigation of pros and cons.

        Path Evaluation:
        Assessing strengths and weaknesses:
        - Branch A: Fast but potentially less accurate
        - Branch B: Accurate but potentially slower
        - Branch C: Best balance but more complex
        - Branch D: Scalable but resource-intensive

        Branch Selection:
        After careful evaluation, Branch C appears most promising for this scenario.

        Synthesis:
        Combining insights from all branches, the optimal solution integrates
        the efficiency of Branch A, accuracy of Branch B, and scalability considerations of Branch D.

        Final Answer: A hybrid approach that balances all key requirements.""",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=150,
        completion_tokens=400,
        total_tokens=550,
        generation_time=6.0,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="complex-response-tot-789",
        metadata={},
    )

    mock_client.generate.return_value = complex_response
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "ToT prompt: {question}"

    reasoning = TreeOfThoughtReasoning(config)
    result = reasoning.execute("What's the best approach for this complex system?")

    # Verify complex analysis
    metrics = result.metadata["approach_specific_metrics"]
    assert metrics["reasoning_branches"] >= 4  # Should detect 4+ branches
    assert metrics["branch_depth"] >= 3  # Should detect high depth exploration
    assert metrics["evaluation_instances"] >= 10  # Should detect many evaluations
    assert metrics["reasoning_quality_score"] > 0.6  # Should be high quality
    assert metrics["contains_problem_decomposition"] is True
    assert metrics["contains_reasoning_branches"] is True
    assert metrics["contains_branch_exploration"] is True
    assert metrics["contains_path_evaluation"] is True
    assert metrics["contains_branch_selection"] is True
    assert metrics["contains_synthesis"] is True


@patch("ml_agents.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_analyze_branches_comprehensive(
    mock_file, mock_create_client, config, mock_client
):
    """Test comprehensive branch analysis."""
    text_with_complex_branches = """
    Alternative 1: Method Alpha
    Option 2: Method Beta
    Path 3: Method Gamma
    Different approach: Method Delta
    Another way: Method Epsilon

    Detailed exploration of each path reveals strengths and weaknesses.
    Comprehensive assessment shows pros and cons for every alternative.
    Most promising approach appears to be Method Gamma.
    """

    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "ToT prompt: {question}"

    reasoning = TreeOfThoughtReasoning(config)
    branch_analysis = reasoning._analyze_branches(text_with_complex_branches)

    assert (
        branch_analysis["reasoning_branches"] >= 5
    )  # Should detect multiple branch indicators
    assert branch_analysis["branch_depth"] >= 3  # Should detect depth indicators
    assert branch_analysis["evaluations"] >= 6  # Should detect evaluation terms
