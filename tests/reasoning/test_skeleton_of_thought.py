"""Tests for Skeleton-of-Thought approach."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.config import ExperimentConfig
from src.reasoning.skeleton_of_thought import SkeletonOfThoughtReasoning
from src.utils.api_clients import StandardResponse


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
        text="Skeleton Outline:\n"
        "1. Problem Analysis\n"
        "2. Solution Framework\n"
        "3. Implementation Strategy\n\n"
        "Section Expansion:\n"
        "1. Problem Analysis: Detailed examination of the issue\n"
        "2. Solution Framework: Systematic approach to resolution\n"
        "3. Implementation Strategy: Step-by-step execution plan\n\n"
        "Progressive Development: Building upon each section systematically\n"
        "Integration: Connecting all components into a cohesive whole\n"
        "Refinement: Polishing the comprehensive solution\n"
        "Final Answer: A well-structured, hierarchical solution.",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=70,
        completion_tokens=140,
        total_tokens=210,
        generation_time=3.5,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="test-response-sot-123",
        metadata={},
    )
    return client


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_init_with_prompt_file(mock_file, mock_create_client, config, mock_client):
    """Test initialization with existing prompt file."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "Test SoT prompt with {question}"

    reasoning = SkeletonOfThoughtReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert reasoning.approach_name == "SkeletonOfThought"
    assert "Test SoT prompt" in reasoning.sot_prompt


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", side_effect=FileNotFoundError)
def test_init_with_fallback_prompt(mock_file, mock_create_client, config, mock_client):
    """Test initialization with fallback prompt when file is missing."""
    mock_create_client.return_value = mock_client

    reasoning = SkeletonOfThoughtReasoning(config)

    assert reasoning.config == config
    assert reasoning.client == mock_client
    assert "outline" in reasoning.sot_prompt


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_basic(mock_file, mock_create_client, config, mock_client):
    """Test basic execution of reasoning approach."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "SoT prompt: {question}"

    reasoning = SkeletonOfThoughtReasoning(config)
    result = reasoning.execute("How do I solve this complex problem?")

    # Verify API client was called
    mock_client.generate.assert_called_once()
    call_args = mock_client.generate.call_args[0][0]
    assert "How do I solve this complex problem?" in call_args

    # Verify response structure
    assert isinstance(result, StandardResponse)
    assert "Skeleton Outline:" in result.text
    assert result.provider == "openrouter"
    assert result.model == "openai/gpt-oss-120b"

    # Verify metadata enhancement
    assert "reasoning_steps" in result.metadata
    assert "approach_specific_metrics" in result.metadata
    assert (
        result.metadata["approach_specific_metrics"]["template_used"]
        == "skeleton_of_thought"
    )


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_metadata_analysis(mock_file, mock_create_client, config, mock_client):
    """Test metadata analysis for structural characteristics."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "SoT prompt: {question}"

    reasoning = SkeletonOfThoughtReasoning(config)
    result = reasoning.execute("Test question")

    metrics = result.metadata["approach_specific_metrics"]

    # Check that all structural components are detected
    assert metrics["contains_skeleton_outline"] is True
    assert metrics["contains_section_expansion"] is True
    assert metrics["contains_progressive_development"] is True
    assert metrics["contains_integration"] is True
    assert metrics["contains_refinement"] is True
    assert metrics["outline_sections"] > 0
    assert metrics["hierarchical_levels"] >= 1
    assert 0.0 <= metrics["structure_quality_score"] <= 1.0


class TestStructuralAnalysis:
    """Test structural analysis methods."""

    @pytest.fixture
    def reasoning_instance(self, config, mock_client):
        """Create reasoning instance for testing."""
        with patch("src.utils.api_clients.create_api_client") as mock_create:
            mock_create.return_value = mock_client
            with patch("builtins.open", side_effect=FileNotFoundError):
                return SkeletonOfThoughtReasoning(config)

    def test_count_outline_sections(self, reasoning_instance):
        """Test counting of outline sections."""
        text_with_sections = """
        1. First main section
        2. Second main section
        3. Third main section
        4. Fourth main section
        """

        sections = reasoning_instance._count_outline_sections(text_with_sections)
        assert sections >= 4  # Should detect multiple sections

    def test_count_outline_sections_bullets(self, reasoning_instance):
        """Test counting with bullet points."""
        text_with_bullets = """
        - First point
        - Second point
        * Third point
        + Fourth point
        """

        sections = reasoning_instance._count_outline_sections(text_with_bullets)
        assert sections >= 4  # Should detect bullet points as sections

    def test_count_outline_sections_minimal(self, reasoning_instance):
        """Test counting with minimal structure content."""
        text_minimal = "This is a simple answer without clear structure."

        sections = reasoning_instance._count_outline_sections(text_minimal)
        assert sections >= 1  # Should return at least 1

    def test_count_hierarchical_levels(self, reasoning_instance):
        """Test counting of hierarchical levels."""
        text_with_hierarchy = """
        1. Main section
            a. Subsection A
            b. Subsection B
        2. Another main section
            a. Subsection C
                i. Sub-subsection
        """

        levels = reasoning_instance._count_hierarchical_levels(text_with_hierarchy)
        assert levels >= 2  # Should detect multiple hierarchy levels

    def test_count_hierarchical_levels_numbered(self, reasoning_instance):
        """Test counting with numbered hierarchies."""
        text_numbered = """
        1.1 First subsection
        1.2 Second subsection
        2.1 Third subsection
        """

        levels = reasoning_instance._count_hierarchical_levels(text_numbered)
        assert levels >= 2  # Should detect numbered hierarchy

    def test_analyze_outline_quality(self, reasoning_instance):
        """Test outline quality analysis."""
        high_quality_text = """
        This response follows a systematic structure and hierarchical organization.
        I will expand each section with detailed development and integrate
        all components systematically into a cohesive framework.
        """

        quality = reasoning_instance._analyze_outline_quality(high_quality_text)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.0  # Should detect structural quality indicators

    def test_analyze_outline_quality_low(self, reasoning_instance):
        """Test outline quality with low-quality text."""
        low_quality_text = "This is a simple answer."

        quality = reasoning_instance._analyze_outline_quality(low_quality_text)
        assert 0.0 <= quality <= 1.0

    def test_has_skeleton_outline(self, reasoning_instance):
        """Test skeleton outline detection."""
        text_with_outline = (
            "Here is the skeleton outline of my approach with main points."
        )
        text_without_outline = "This is just a regular response."

        assert reasoning_instance._has_skeleton_outline(text_with_outline) is True
        assert reasoning_instance._has_skeleton_outline(text_without_outline) is False

    def test_has_section_expansion(self, reasoning_instance):
        """Test section expansion detection."""
        text_with_expansion = "I will expand each section with detailed elaboration."
        text_without_expansion = "This is just a regular response."

        assert reasoning_instance._has_section_expansion(text_with_expansion) is True
        assert (
            reasoning_instance._has_section_expansion(text_without_expansion) is False
        )

    def test_has_progressive_development(self, reasoning_instance):
        """Test progressive development detection."""
        text_with_development = (
            "Building upon each section progressively and systematically."
        )
        text_without_development = "This is just a regular response."

        assert (
            reasoning_instance._has_progressive_development(text_with_development)
            is True
        )
        assert (
            reasoning_instance._has_progressive_development(text_without_development)
            is False
        )

    def test_has_integration(self, reasoning_instance):
        """Test integration detection."""
        text_with_integration = "I will integrate all components into a cohesive whole."
        text_without_integration = "This is just a regular response."

        assert reasoning_instance._has_integration(text_with_integration) is True
        assert reasoning_instance._has_integration(text_without_integration) is False

    def test_has_refinement(self, reasoning_instance):
        """Test refinement detection."""
        text_with_refinement = (
            "Finally, I will refine and polish the comprehensive solution."
        )
        text_without_refinement = "This is just a regular response."

        assert reasoning_instance._has_refinement(text_with_refinement) is True
        assert reasoning_instance._has_refinement(text_without_refinement) is False


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_execute_error_handling(mock_file, mock_create_client, config):
    """Test error handling during execution."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = Exception("API Error")
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "SoT prompt: {question}"

    reasoning = SkeletonOfThoughtReasoning(config)

    with pytest.raises(Exception, match="API Error"):
        reasoning.execute("Test question")


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_inheritance_from_base(mock_file, mock_create_client, config, mock_client):
    """Test that the class properly inherits from BaseReasoning."""
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "SoT prompt: {question}"

    reasoning = SkeletonOfThoughtReasoning(config)

    # Should have inherited methods from BaseReasoning
    assert hasattr(reasoning, "_enhance_metadata")
    assert hasattr(reasoning, "execute")
    assert reasoning.approach_name == "SkeletonOfThought"


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_complex_structural_response(
    mock_file, mock_create_client, config, mock_client
):
    """Test with a complex structural response."""
    complex_response = StandardResponse(
        text="""Skeleton Outline:
        I. Problem Understanding
        II. Analysis Framework
        III. Solution Design
        IV. Implementation Plan
        V. Evaluation Criteria

        Section Expansion:
        I. Problem Understanding
           A. Define the core issue
           B. Identify constraints
           C. Determine success criteria

        II. Analysis Framework
           A. Data collection methods
           B. Analysis techniques
           C. Decision criteria

        Progressive Development:
        Building systematically upon each component, I will develop
        a comprehensive framework that integrates all elements.

        Integration:
        All sections connect to form a cohesive methodology that
        addresses the problem holistically.

        Refinement:
        The final solution is polished and comprehensive, ensuring
        all aspects are thoroughly addressed.

        Final Answer: A structured, hierarchical approach that
        systematically addresses all aspects of the complex problem.""",
        provider="openrouter",
        model="openai/gpt-oss-120b",
        prompt_tokens=120,
        completion_tokens=280,
        total_tokens=400,
        generation_time=5.0,
        parameters={"temperature": 0.3, "max_tokens": 512},
        response_id="complex-response-sot-456",
        metadata={},
    )

    mock_client.generate.return_value = complex_response
    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "SoT prompt: {question}"

    reasoning = SkeletonOfThoughtReasoning(config)
    result = reasoning.execute("How do I approach a complex multi-faceted problem?")

    # Verify complex analysis
    metrics = result.metadata["approach_specific_metrics"]
    assert (
        metrics["outline_sections"] >= 5
    )  # Should detect Roman numerals and sub-items
    assert (
        metrics["hierarchical_levels"] >= 2
    )  # Should detect multiple hierarchy levels
    assert metrics["structure_quality_score"] > 0.5  # Should be high quality
    assert metrics["contains_skeleton_outline"] is True
    assert metrics["contains_section_expansion"] is True
    assert metrics["contains_progressive_development"] is True
    assert metrics["contains_integration"] is True
    assert metrics["contains_refinement"] is True


@patch("src.utils.api_clients.create_api_client")
@patch("builtins.open", new_callable=mock_open)
def test_analyze_structure_comprehensive(
    mock_file, mock_create_client, config, mock_client
):
    """Test comprehensive structure analysis."""
    complex_structural_text = """
    A. Main Section One
       1. Subsection 1.1
       2. Subsection 1.2
          a. Sub-subsection
          b. Another sub-subsection
    B. Main Section Two
       1. Subsection 2.1
    """

    mock_create_client.return_value = mock_client
    mock_file.return_value.read.return_value = "SoT prompt: {question}"

    reasoning = SkeletonOfThoughtReasoning(config)
    structure = reasoning._analyze_structure(complex_structural_text)

    assert structure["outline_sections"] >= 6  # Should count multiple types of sections
    assert (
        structure["hierarchical_levels"] >= 3
    )  # Should detect multiple indentation levels
