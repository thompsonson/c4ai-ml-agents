"""Tests for the Chain-of-Thought reasoning approach."""

from unittest.mock import Mock, mock_open, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.chain_of_thought import ChainOfThoughtReasoning
from ml_agents.utils.api_clients import StandardResponse


class TestChainOfThoughtReasoning:
    """Test cases for Chain-of-Thought reasoning approach."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="test_provider", model="test-model", temperature=0.3
        )

    @pytest.fixture
    def mock_client(self):
        """Create mock API client."""
        client = Mock()
        return client

    @pytest.fixture
    def cot_prompt_content(self):
        """Sample CoT prompt template."""
        return """Think step by step:

Question: {question}

Please work through this systematically."""

    def create_cot_response(
        self,
        text="Step 1: First I need to understand the problem.\nStep 2: Then I analyze the data.\nTherefore, the answer is 42.",
    ):
        """Helper to create a CoT-style response."""
        return StandardResponse(
            text=text,
            provider="test_provider",
            model="test-model",
            prompt_tokens=25,
            completion_tokens=35,
            total_tokens=60,
            generation_time=2.0,
            parameters={"temperature": 0.3},
            response_id="cot-123",
            metadata={},
        )

    @patch("ml_agents.gents.reasoning.chain_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialization_with_prompt_file(
        self, mock_file, mock_create_client, config, mock_client, cot_prompt_content
    ):
        """Test CoT initialization with prompt file."""
        mock_file.return_value.read.return_value = cot_prompt_content
        mock_create_client.return_value = mock_client

        reasoning = ChainOfThoughtReasoning(config)

        assert reasoning.config == config
        assert reasoning.client == mock_client
        assert reasoning.approach_name == "ChainOfThought"
        assert "Think step by step" in reasoning.cot_prompt

    @patch("ml_agents.gents.reasoning.chain_of_thought.create_api_client")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_initialization_without_prompt_file(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test CoT initialization when prompt file is missing."""
        mock_create_client.return_value = mock_client

        reasoning = ChainOfThoughtReasoning(config)

        # Should use fallback prompt
        assert "Please think through this step by step" in reasoning.cot_prompt
        assert "{question}" in reasoning.cot_prompt

    @patch("ml_agents.gents.reasoning.chain_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_basic(
        self, mock_file, mock_create_client, config, mock_client, cot_prompt_content
    ):
        """Test basic execution of CoT reasoning."""
        mock_file.return_value.read.return_value = cot_prompt_content
        mock_create_client.return_value = mock_client

        cot_response = self.create_cot_response()
        mock_client.generate.return_value = cot_response

        reasoning = ChainOfThoughtReasoning(config)
        test_prompt = "What is 2 + 2?"

        result = reasoning.execute(test_prompt)

        # Verify API client was called with CoT-enhanced prompt
        expected_prompt = f"Think step by step:\n\nQuestion: {test_prompt}\n\nPlease work through this systematically."
        mock_client.generate.assert_called_once_with(expected_prompt)

        # Verify result structure
        assert isinstance(result, StandardResponse)
        assert result.metadata["reasoning_approach"] == "ChainOfThought"
        assert result.metadata["reasoning_steps"] > 0

    @patch("ml_agents.gents.reasoning.chain_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_metadata_analysis(
        self, mock_file, mock_create_client, config, mock_client, cot_prompt_content
    ):
        """Test metadata analysis in CoT response."""
        mock_file.return_value.read.return_value = cot_prompt_content
        mock_create_client.return_value = mock_client

        # Create response with clear CoT patterns
        cot_text = """Step 1: First, I need to understand what the question is asking.
Step 2: Then, I'll break down the problem into smaller parts.
Therefore, I can conclude that the answer is 42.
This means we have found the solution."""

        cot_response = self.create_cot_response(cot_text)
        mock_client.generate.return_value = cot_response

        reasoning = ChainOfThoughtReasoning(config)
        result = reasoning.execute("Test question")

        # Verify CoT-specific metadata
        assert result.metadata["reasoning_steps"] >= 2
        assert "step_quality_score" in result.metadata["approach_specific_metrics"]
        assert "contains_numbered_steps" in result.metadata["approach_specific_metrics"]
        assert (
            "contains_logical_connectors"
            in result.metadata["approach_specific_metrics"]
        )
        assert (
            result.metadata["approach_specific_metrics"]["template_used"]
            == "chain_of_thought"
        )

    def test_count_cot_steps(self):
        """Test CoT-specific step counting."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.chain_of_thought.create_api_client"):
            reasoning = ChainOfThoughtReasoning(config)

            # Test numbered steps
            text1 = (
                "Step 1: First analysis\nStep 2: Second analysis\nStep 3: Conclusion"
            )
            steps1 = reasoning._count_cot_steps(text1)
            assert steps1 >= 3

            # Test numbered lists
            text2 = "1. First point\n2. Second point\n3. Third point"
            steps2 = reasoning._count_cot_steps(text2)
            assert steps2 >= 3

            # Test sequence words
            text3 = "First, we analyze. Then, we evaluate. Finally, we conclude."
            steps3 = reasoning._count_cot_steps(text3)
            assert steps3 >= 3

            # Test no clear steps
            text4 = "This is just a plain response without clear reasoning steps."
            steps4 = reasoning._count_cot_steps(text4)
            assert steps4 == 1  # Should return at least 1

    def test_analyze_step_quality(self):
        """Test step quality analysis."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.chain_of_thought.create_api_client"):
            reasoning = ChainOfThoughtReasoning(config)

            # High quality text with logical connectors
            high_quality = """
            I need to analyze this problem because it's complex.
            This means I should break it down systematically.
            Therefore, I'll examine each component carefully.
            This shows that a structured approach is necessary.
            """
            quality1 = reasoning._analyze_step_quality(high_quality)
            assert quality1 > 0.0

            # Low quality text
            low_quality = "Simple answer without reasoning indicators."
            quality2 = reasoning._analyze_step_quality(low_quality)
            assert quality2 <= quality1  # Should be lower quality

    def test_has_numbered_steps(self):
        """Test numbered steps detection."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.chain_of_thought.create_api_client"):
            reasoning = ChainOfThoughtReasoning(config)

            # Text with numbered steps
            text_with_numbers = "1. First step\n2. Second step\n3. Third step"
            assert reasoning._has_numbered_steps(text_with_numbers) is True

            # Text without numbered steps
            text_without_numbers = "No numbered steps in this text."
            assert reasoning._has_numbered_steps(text_without_numbers) is False

            # Text with only one number (should return False)
            text_one_number = "1. Only one numbered item here."
            assert reasoning._has_numbered_steps(text_one_number) is False

    def test_has_logical_connectors(self):
        """Test logical connectors detection."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.chain_of_thought.create_api_client"):
            reasoning = ChainOfThoughtReasoning(config)

            # Text with logical connectors
            text_with_connectors = "Because of this analysis, therefore I conclude that the answer is correct."
            assert reasoning._has_logical_connectors(text_with_connectors) is True

            # Text without logical connectors
            text_without_connectors = (
                "Simple statement without logical reasoning words."
            )
            assert reasoning._has_logical_connectors(text_without_connectors) is False

    @patch("ml_agents.gents.reasoning.chain_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_with_complex_reasoning(
        self, mock_file, mock_create_client, config, mock_client, cot_prompt_content
    ):
        """Test execution with complex reasoning patterns."""
        mock_file.return_value.read.return_value = cot_prompt_content
        mock_create_client.return_value = mock_client

        # Create response with rich CoT patterns
        complex_response_text = """
        Step 1: First, I need to understand the problem clearly.
        Step 2: Then, I'll analyze the given information systematically.
        Step 3: Next, I'll identify the key relationships.

        Because the problem involves multiple variables, I need to consider each one carefully.
        Therefore, I'll examine each component:

        1. Component A shows pattern X
        2. Component B indicates trend Y
        3. Component C suggests outcome Z

        This means that when we combine these factors, we get a clear picture.
        Hence, the final answer is 42, as this shows the logical conclusion.
        """

        complex_response = self.create_cot_response(complex_response_text)
        mock_client.generate.return_value = complex_response

        reasoning = ChainOfThoughtReasoning(config)
        result = reasoning.execute("Complex problem analysis")

        # Verify enhanced metadata
        assert result.metadata["reasoning_steps"] > 5  # Should detect many steps
        assert result.metadata["approach_specific_metrics"]["step_quality_score"] > 0.1
        assert (
            result.metadata["approach_specific_metrics"]["contains_numbered_steps"]
            is True
        )
        assert (
            result.metadata["approach_specific_metrics"]["contains_logical_connectors"]
            is True
        )

    @patch("ml_agents.gents.reasoning.chain_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_performance_tracking(
        self, mock_file, mock_create_client, config, mock_client, cot_prompt_content
    ):
        """Test performance tracking in CoT execution."""
        mock_file.return_value.read.return_value = cot_prompt_content
        mock_create_client.return_value = mock_client

        cot_response = self.create_cot_response()
        mock_client.generate.return_value = cot_response

        reasoning = ChainOfThoughtReasoning(config)
        result = reasoning.execute("Performance test")

        # Verify performance metrics are preserved
        assert result.generation_time == 2.0
        assert result.total_tokens == 60
        assert (
            result.metadata["approach_specific_metrics"]["original_prompt"]
            == "Performance test"
        )

    @patch("ml_agents.gents.reasoning.chain_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_multiple_executions_consistency(
        self, mock_file, mock_create_client, config, mock_client, cot_prompt_content
    ):
        """Test consistency across multiple executions."""
        mock_file.return_value.read.return_value = cot_prompt_content
        mock_create_client.return_value = mock_client

        reasoning = ChainOfThoughtReasoning(config)

        # Execute multiple times with different responses
        test_cases = [
            ("Simple question", "Step 1: Analyze\nTherefore: Answer"),
            (
                "Complex question",
                "Step 1: Break down\nStep 2: Evaluate\nStep 3: Conclude",
            ),
            (
                "Mathematical question",
                "1. Identify variables\n2. Apply formula\n3. Calculate result",
            ),
        ]

        for prompt, response_text in test_cases:
            mock_client.generate.return_value = self.create_cot_response(response_text)
            result = reasoning.execute(prompt)

            # Verify consistent metadata structure
            assert result.metadata["reasoning_approach"] == "ChainOfThought"
            assert "reasoning_steps" in result.metadata
            assert "approach_specific_metrics" in result.metadata
            assert (
                result.metadata["approach_specific_metrics"]["original_prompt"]
                == prompt
            )
