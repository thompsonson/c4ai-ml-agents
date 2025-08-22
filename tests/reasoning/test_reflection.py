"""Tests for the Reflection reasoning approach."""

from unittest.mock import Mock, mock_open, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.reflection import ReflectionReasoning
from ml_agents.utils.api_clients import StandardResponse


class TestReflectionReasoning:
    """Test cases for Reflection reasoning approach."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="test_provider", model="test-model", temperature=0.3
        )

    @pytest.fixture
    def multi_step_config(self):
        """Create test configuration with multi-step enabled."""
        config = ExperimentConfig(
            provider="test_provider", model="test-model", temperature=0.3
        )
        config.multi_step_reflection = True
        config.max_reflection_iterations = 2
        config.reflection_threshold = 0.7
        return config

    @pytest.fixture
    def mock_client(self):
        """Create mock API client."""
        client = Mock()
        return client

    @pytest.fixture
    def reflection_prompt_content(self):
        """Sample Reflection prompt template."""
        return """Think and reflect:

Question: {question}

**Initial Response:**
Provide your first answer.

**Reflection:**
Examine your response critically.

**Refined Response:**
Improve your answer."""

    def create_reflection_response(self, text=None):
        """Helper to create a Reflection-style response."""
        if text is None:
            text = """**Initial Response:**
The answer is probably 42 based on my first impression.

**Reflection:**
Let me think more carefully about this. I made some assumptions that might not be correct.
I should consider alternative approaches and verify my reasoning.

**Refined Response:**
After reflection, I realize the answer requires more nuanced analysis. The correct answer is 24."""

        return StandardResponse(
            text=text,
            provider="test_provider",
            model="test-model",
            prompt_tokens=40,
            completion_tokens=60,
            total_tokens=100,
            generation_time=3.0,
            parameters={"temperature": 0.3},
            response_id="reflection-123",
            metadata={},
        )

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialization_with_prompt_file(
        self,
        mock_file,
        mock_create_client,
        config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test Reflection initialization with prompt file."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client

        reasoning = ReflectionReasoning(config)

        assert reasoning.config == config
        assert reasoning.client == mock_client
        assert reasoning.approach_name == "Reflection"
        assert "Think and reflect" in reasoning.reflection_prompt
        assert reasoning.max_iterations == 2  # Default value

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_initialization_without_prompt_file(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test Reflection initialization when prompt file is missing."""
        mock_create_client.return_value = mock_client

        reasoning = ReflectionReasoning(config)

        # Should use fallback prompt
        assert (
            "Please answer this question, then reflect on your answer and improve it"
            in reasoning.reflection_prompt
        )
        assert "{question}" in reasoning.reflection_prompt

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_single_prompt_mode(
        self,
        mock_file,
        mock_create_client,
        config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test basic execution of Reflection reasoning (single prompt mode)."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client

        reflection_response = self.create_reflection_response()
        mock_client.generate.return_value = reflection_response

        reasoning = ReflectionReasoning(config)
        test_prompt = "What is the best approach to solve this?"

        result = reasoning.execute(test_prompt)

        # Verify API client was called once with structured prompt
        assert mock_client.generate.call_count == 1

        # Verify result structure
        assert isinstance(result, StandardResponse)
        assert result.metadata["reasoning_approach"] == "Reflection"
        assert result.metadata["reasoning_steps"] >= 2  # Initial + reflection sections

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_metadata_analysis(
        self,
        mock_file,
        mock_create_client,
        config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test metadata analysis in Reflection response."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client

        # Create response with clear reflection patterns
        reflection_text = """**Initial Response:**
My initial answer is X because of reason A.

**Reflection:**
Upon reflection, I realize I need to reconsider my assumptions.
Let me examine this more critically and improve my reasoning.
I notice several issues with my initial approach.

**Refined Response:**
After careful reflection, the better answer is Y because of improved reasoning."""

        reflection_response = self.create_reflection_response(reflection_text)
        mock_client.generate.return_value = reflection_response

        reasoning = ReflectionReasoning(config)
        result = reasoning.execute("Test question")

        # Verify Reflection-specific metadata
        assert (
            "reflection_quality_score" in result.metadata["approach_specific_metrics"]
        )
        assert "has_initial_response" in result.metadata["approach_specific_metrics"]
        assert "has_reflection_section" in result.metadata["approach_specific_metrics"]
        assert "has_refined_response" in result.metadata["approach_specific_metrics"]
        assert (
            result.metadata["approach_specific_metrics"]["template_used"]
            == "reflection"
        )

        # Verify intermediate results are captured
        assert "intermediate_results" in result.metadata
        assert "reflection_sections" in result.metadata["intermediate_results"]

    def test_parse_reflection_structure(self):
        """Test reflection structure parsing."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.reflection.create_api_client"):
            reasoning = ReflectionReasoning(config)

            # Text with clear reflection structure
            structured_text = """**Initial Response:**
My first attempt at the answer.

**Reflection:**
Let me reconsider this approach. I made some errors.

**Refined Response:**
Here's my improved answer after reflection."""

            structure = reasoning._parse_reflection_structure(structured_text)

            assert structure["has_initial"] is True
            assert structure["has_reflection"] is True
            assert structure["has_refined"] is True
            assert structure["improvement_count"] > 0
            assert len(structure["sections"]) > 0

    def test_analyze_reflection_quality(self):
        """Test reflection quality analysis."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.reflection.create_api_client"):
            reasoning = ReflectionReasoning(config)

            # High quality reflection text
            high_quality = """
            Looking at my response, I need to analyze what I said more carefully.
            My reasoning approach could be improved by considering alternative viewpoints.
            Let me examine the logic and enhance my thinking process.
            This will help me provide a better answer.
            """
            quality1 = reasoning._analyze_reflection_quality(high_quality)
            assert quality1 > 0.0

            # Low quality text without reflection indicators
            low_quality = "Simple answer without any self-examination."
            quality2 = reasoning._analyze_reflection_quality(low_quality)
            assert quality2 <= quality1

    def test_count_reflection_steps(self):
        """Test reflection step counting."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.reflection.create_api_client"):
            reasoning = ReflectionReasoning(config)

            # Text with reflection sections and actions
            reflection_text = """Initial Response: First answer
            Reflection: Let me reconsider and examine this more carefully.
            I need to improve my approach and refine the logic.
            Refined Response: Better answer after reflection."""

            steps = reasoning._count_reflection_steps(reflection_text)
            assert steps >= 2  # At minimum initial + refined

    def test_has_structured_reflection(self):
        """Test structured reflection detection."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.reflection.create_api_client"):
            reasoning = ReflectionReasoning(config)

            # Text with clear structure markers
            structured = """**Initial Response:**
            First answer
            **Reflection:**
            Critical analysis
            **Refined Response:**
            Improved answer"""

            assert reasoning._has_structured_reflection(structured) is True

            # Text without clear structure
            unstructured = "Just a plain response without reflection markers."
            assert reasoning._has_structured_reflection(unstructured) is False

    def test_extract_reflection_sections(self):
        """Test reflection section extraction."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.reflection.create_api_client"):
            reasoning = ReflectionReasoning(config)

            text = """**Initial Response:** First attempt
            **Reflection:** Critical analysis
            **Refined Response:** Improved version"""

            sections = reasoning._extract_reflection_sections(text)

            assert "initial_response" in sections
            assert "reflection" in sections
            assert "refined_response" in sections

    def test_extract_issues_and_improvements(self):
        """Test issue and improvement extraction."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.reflection.create_api_client"):
            reasoning = ReflectionReasoning(config)

            # Text mentioning issues and improvements
            text = """I made an error in my initial approach. This was incomplete and wrong.
            I need to improve this by being more careful and adding missing details."""

            issues = reasoning._extract_issues(text)
            improvements = reasoning._extract_improvements(text)

            assert len(issues) > 0
            assert len(improvements) > 0
            assert "error" in issues
            assert "improve" in improvements

    # Multi-step mode tests would be added after implementing the multi-step functionality
    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_multi_step_reflection_flag(
        self,
        mock_file,
        mock_create_client,
        multi_step_config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test that multi-step flag is recognized in configuration."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client

        reflection_response = self.create_reflection_response()
        mock_client.generate.return_value = reflection_response

        reasoning = ReflectionReasoning(multi_step_config)

        # Verify configuration parameters are available
        assert hasattr(multi_step_config, "multi_step_reflection")
        assert multi_step_config.multi_step_reflection is True
        assert reasoning.max_iterations == 2
        assert reasoning.reflection_threshold == 0.7

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_performance_tracking(
        self,
        mock_file,
        mock_create_client,
        config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test performance tracking in Reflection execution."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client

        reflection_response = self.create_reflection_response()
        mock_client.generate.return_value = reflection_response

        reasoning = ReflectionReasoning(config)
        result = reasoning.execute("Performance test")

        # Verify performance metrics are preserved
        assert result.generation_time == 3.0
        assert result.total_tokens == 100
        assert (
            result.metadata["approach_specific_metrics"]["original_prompt"]
            == "Performance test"
        )

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_without_clear_structure(
        self,
        mock_file,
        mock_create_client,
        config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test execution with response that lacks clear reflection structure."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client

        # Response without clear structure markers
        unstructured_response = self.create_reflection_response(
            "This is just a regular response without clear initial/reflection/refined sections."
        )
        mock_client.generate.return_value = unstructured_response

        reasoning = ReflectionReasoning(config)
        result = reasoning.execute("Test without structure")

        # Should still work but with lower structure scores
        assert (
            result.metadata["approach_specific_metrics"]["has_initial_response"]
            is False
        )
        assert (
            result.metadata["approach_specific_metrics"]["has_reflection_section"]
            is False
        )
        assert (
            result.metadata["approach_specific_metrics"]["has_refined_response"]
            is False
        )

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_multiple_executions_consistency(
        self,
        mock_file,
        mock_create_client,
        config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test consistency across multiple executions."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client

        reasoning = ReflectionReasoning(config)

        # Execute multiple times with different reflection patterns
        test_cases = [
            ("Simple question", "Initial: A. Reflection: Reconsider. Refined: B."),
            (
                "Complex question",
                "**Initial Response:** X **Reflection:** Critical analysis **Refined Response:** Y",
            ),
            (
                "Detailed question",
                "First attempt: Z. Upon reflection, I need to improve this. Better answer: W.",
            ),
        ]

        for prompt, response_text in test_cases:
            mock_client.generate.return_value = self.create_reflection_response(
                response_text
            )
            result = reasoning.execute(prompt)

            # Verify consistent metadata structure
            assert result.metadata["reasoning_approach"] == "Reflection"
            assert "reasoning_steps" in result.metadata
            assert "approach_specific_metrics" in result.metadata
            assert "intermediate_results" in result.metadata
            assert (
                result.metadata["approach_specific_metrics"]["original_prompt"]
                == prompt
            )

    @patch("ml_agents.reasoning.reflection.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_cleanup(
        self,
        mock_file,
        mock_create_client,
        config,
        mock_client,
        reflection_prompt_content,
    ):
        """Test cleanup functionality."""
        mock_file.return_value.read.return_value = reflection_prompt_content
        mock_create_client.return_value = mock_client
        mock_client.cleanup = Mock()

        reasoning = ReflectionReasoning(config)
        reasoning.cleanup()

        mock_client.cleanup.assert_called_once()

    # Placeholder tests for future multi-step implementation
    def test_multi_step_execution_placeholder(self):
        """Placeholder test for multi-step execution (to be implemented)."""
        # This test will be completed when multi-step reflection is implemented
        # It should test:
        # - Multiple API calls (initial -> reflection -> refinement)
        # - Iteration limits (max_iterations)
        # - Quality threshold checking (reflection_threshold)
        # - Intermediate result storage
        pass

    def test_needs_refinement_placeholder(self):
        """Placeholder test for refinement decision logic (to be implemented)."""
        # This test will verify the _needs_refinement method when implemented
        # It should test quality threshold evaluation
        pass
