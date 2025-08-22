"""Tests for the Program-of-Thought reasoning approach."""

from unittest.mock import Mock, mock_open, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.program_of_thought import ProgramOfThoughtReasoning
from ml_agents.utils.api_clients import StandardResponse


class TestProgramOfThoughtReasoning:
    """Test cases for Program-of-Thought reasoning approach."""

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
    def pot_prompt_content(self):
        """Sample PoT prompt template."""
        return """Solve this with code:

Question: {question}

Write Python code to solve this step by step."""

    def create_pot_response(self, text=None):
        """Helper to create a PoT-style response with code."""
        if text is None:
            text = """Let me solve this with Python code:

```python
# Calculate the result
def calculate():
    x = 10
    y = 20
    return x + y

result = calculate()
print(f"The answer is {result}")
```

The answer is 30."""

        return StandardResponse(
            text=text,
            provider="test_provider",
            model="test-model",
            prompt_tokens=30,
            completion_tokens=50,
            total_tokens=80,
            generation_time=2.5,
            parameters={"temperature": 0.3},
            response_id="pot-123",
            metadata={},
        )

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialization_with_prompt_file(
        self, mock_file, mock_create_client, config, mock_client, pot_prompt_content
    ):
        """Test PoT initialization with prompt file."""
        mock_file.return_value.read.return_value = pot_prompt_content
        mock_create_client.return_value = mock_client

        reasoning = ProgramOfThoughtReasoning(config)

        assert reasoning.config == config
        assert reasoning.client == mock_client
        assert reasoning.approach_name == "ProgramOfThought"
        assert "Solve this with code" in reasoning.pot_prompt

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_initialization_without_prompt_file(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test PoT initialization when prompt file is missing."""
        mock_create_client.return_value = mock_client

        reasoning = ProgramOfThoughtReasoning(config)

        # Should use fallback prompt
        assert (
            "Let's solve this problem step by step using programming logic"
            in reasoning.pot_prompt
        )
        assert "{question}" in reasoning.pot_prompt

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_basic(
        self, mock_file, mock_create_client, config, mock_client, pot_prompt_content
    ):
        """Test basic execution of PoT reasoning."""
        mock_file.return_value.read.return_value = pot_prompt_content
        mock_create_client.return_value = mock_client

        pot_response = self.create_pot_response()
        mock_client.generate.return_value = pot_response

        reasoning = ProgramOfThoughtReasoning(config)
        test_prompt = "Calculate 10 + 20"

        result = reasoning.execute(test_prompt)

        # Verify API client was called with PoT-enhanced prompt
        expected_prompt = f"Solve this with code:\n\nQuestion: {test_prompt}\n\nWrite Python code to solve this step by step."
        mock_client.generate.assert_called_once_with(expected_prompt)

        # Verify result structure
        assert isinstance(result, StandardResponse)
        assert result.metadata["reasoning_approach"] == "ProgramOfThought"
        assert result.metadata["reasoning_steps"] > 0

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_metadata_analysis(
        self, mock_file, mock_create_client, config, mock_client, pot_prompt_content
    ):
        """Test metadata analysis in PoT response."""
        mock_file.return_value.read.return_value = pot_prompt_content
        mock_create_client.return_value = mock_client

        # Create response with clear PoT patterns
        pot_text = """Let me solve this computationally:

```python
def solve_problem():
    # Define variables
    numbers = [1, 2, 3, 4, 5]

    # Process the data
    result = sum(numbers)

    return result

answer = solve_problem()
print(f"Result: {answer}")
```

This code calculates the sum as 15."""

        pot_response = self.create_pot_response(pot_text)
        mock_client.generate.return_value = pot_response

        reasoning = ProgramOfThoughtReasoning(config)
        result = reasoning.execute("Sum the numbers 1 through 5")

        # Verify PoT-specific metadata
        assert result.metadata["reasoning_steps"] >= 1
        assert "code_blocks_count" in result.metadata["approach_specific_metrics"]
        assert (
            "programming_quality_score" in result.metadata["approach_specific_metrics"]
        )
        assert "contains_variables" in result.metadata["approach_specific_metrics"]
        assert "contains_functions" in result.metadata["approach_specific_metrics"]
        assert (
            result.metadata["approach_specific_metrics"]["template_used"]
            == "program_of_thought"
        )

    def test_count_code_blocks(self):
        """Test code block counting functionality."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.program_of_thought.create_api_client"):
            reasoning = ProgramOfThoughtReasoning(config)

            # Test markdown code blocks
            text1 = """```python
def test():
    return 42
```"""
            count1 = reasoning._count_code_blocks(text1)
            assert count1 >= 1

            # Test inline code
            text2 = "Use `variable = value` to assign values"
            count2 = reasoning._count_code_blocks(text2)
            assert count2 >= 1

            # Test variable assignments
            text3 = """x = 10
y = 20
result = x + y"""
            count3 = reasoning._count_code_blocks(text3)
            assert count3 >= 3  # Three assignments

            # Test function definitions
            text4 = "def calculate(): return 42"
            count4 = reasoning._count_code_blocks(text4)
            assert count4 >= 1

    def test_analyze_programming_quality(self):
        """Test programming quality analysis."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.program_of_thought.create_api_client"):
            reasoning = ProgramOfThoughtReasoning(config)

            # High quality programming text
            high_quality = """
            Let me define a function to solve this algorithmically.
            I'll iterate through the data and calculate the result.
            This code processes the input systematically.
            The function returns the computed value.
            """
            quality1 = reasoning._analyze_programming_quality(high_quality)
            assert quality1 > 0.0

            # Low quality text without programming concepts
            low_quality = "Simple answer without code or programming concepts."
            quality2 = reasoning._analyze_programming_quality(low_quality)
            assert quality2 <= quality1

    def test_count_computational_steps(self):
        """Test computational step counting."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.program_of_thought.create_api_client"):
            reasoning = ProgramOfThoughtReasoning(config)

            # Text with code blocks and steps
            text_with_steps = """Step 1: Define variables
```python
x = 10
y = 20
```

Step 2: Calculate result
```python
result = x + y
```"""

            steps = reasoning._count_computational_steps(text_with_steps)
            assert steps >= 4  # 2 steps + 2 code blocks + 2 assignments

    def test_has_variables(self):
        """Test variable detection."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.program_of_thought.create_api_client"):
            reasoning = ProgramOfThoughtReasoning(config)

            # Text with variable assignments
            text_with_vars = "Let me set x = 10 and y = 20 for this calculation."
            assert reasoning._has_variables(text_with_vars) is True

            # Text without variables
            text_without_vars = "This is just plain text without assignments."
            assert reasoning._has_variables(text_without_vars) is False

    def test_has_functions(self):
        """Test function detection."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.program_of_thought.create_api_client"):
            reasoning = ProgramOfThoughtReasoning(config)

            # Text with function definition
            text_with_functions = "def calculate(): return 42"
            assert reasoning._has_functions(text_with_functions) is True

            # Text with function calls
            text_with_calls = "result = calculate(10, 20)"
            assert reasoning._has_functions(text_with_calls) is True

            # Text without functions
            text_without_functions = "Simple calculation without function usage."
            assert reasoning._has_functions(text_without_functions) is False

    def test_has_control_structures(self):
        """Test control structure detection."""
        config = ExperimentConfig()

        with patch("ml_agents.reasoning.program_of_thought.create_api_client"):
            reasoning = ProgramOfThoughtReasoning(config)

            # Text with control structures
            text_with_control = (
                "if condition: do something, else iterate through the loop"
            )
            assert reasoning._has_control_structures(text_with_control) is True

            # Text without control structures
            text_without_control = "Simple assignment operation."
            assert reasoning._has_control_structures(text_without_control) is False

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_with_complex_code(
        self, mock_file, mock_create_client, config, mock_client, pot_prompt_content
    ):
        """Test execution with complex programming patterns."""
        mock_file.return_value.read.return_value = pot_prompt_content
        mock_create_client.return_value = mock_client

        # Create response with rich programming patterns
        complex_code_text = """
        Let me solve this algorithmically:

        ```python
        def analyze_data(numbers):
            # Initialize variables
            total = 0
            count = 0

            # Iterate through the data
            for num in numbers:
                if num > 0:  # Filter positive numbers
                    total += num
                    count += 1

            # Calculate average
            average = total / count if count > 0 else 0
            return average

        # Process the input
        data = [1, 2, 3, 4, 5]
        result = analyze_data(data)
        print(f"Average: {result}")
        ```

        This code computes the average systematically by iterating through the data
        and applying conditional logic to filter the values.
        """

        complex_response = self.create_pot_response(complex_code_text)
        mock_client.generate.return_value = complex_response

        reasoning = ProgramOfThoughtReasoning(config)
        result = reasoning.execute("Calculate the average of positive numbers")

        # Verify enhanced metadata for complex code
        assert result.metadata["reasoning_steps"] > 3
        assert result.metadata["approach_specific_metrics"]["code_blocks_count"] > 0
        assert (
            result.metadata["approach_specific_metrics"]["programming_quality_score"]
            > 0.1
        )
        assert (
            result.metadata["approach_specific_metrics"]["contains_variables"] is True
        )
        assert (
            result.metadata["approach_specific_metrics"]["contains_functions"] is True
        )
        assert (
            result.metadata["approach_specific_metrics"]["contains_loops_conditions"]
            is True
        )

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_execute_performance_tracking(
        self, mock_file, mock_create_client, config, mock_client, pot_prompt_content
    ):
        """Test performance tracking in PoT execution."""
        mock_file.return_value.read.return_value = pot_prompt_content
        mock_create_client.return_value = mock_client

        pot_response = self.create_pot_response()
        mock_client.generate.return_value = pot_response

        reasoning = ProgramOfThoughtReasoning(config)
        result = reasoning.execute("Performance test")

        # Verify performance metrics are preserved
        assert result.generation_time == 2.5
        assert result.total_tokens == 80
        assert (
            result.metadata["approach_specific_metrics"]["original_prompt"]
            == "Performance test"
        )

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_multiple_executions_consistency(
        self, mock_file, mock_create_client, config, mock_client, pot_prompt_content
    ):
        """Test consistency across multiple executions."""
        mock_file.return_value.read.return_value = pot_prompt_content
        mock_create_client.return_value = mock_client

        reasoning = ProgramOfThoughtReasoning(config)

        # Execute multiple times with different code patterns
        test_cases = [
            ("Math problem", "x = 5\ny = 10\nresult = x * y"),
            ("Data processing", "def process(): return [1,2,3]"),
            ("Algorithm", "for i in range(10): print(i)"),
        ]

        for prompt, response_text in test_cases:
            mock_client.generate.return_value = self.create_pot_response(response_text)
            result = reasoning.execute(prompt)

            # Verify consistent metadata structure
            assert result.metadata["reasoning_approach"] == "ProgramOfThought"
            assert "reasoning_steps" in result.metadata
            assert "approach_specific_metrics" in result.metadata
            assert (
                result.metadata["approach_specific_metrics"]["original_prompt"]
                == prompt
            )

    @patch("ml_agents.reasoning.program_of_thought.create_api_client")
    @patch("builtins.open", new_callable=mock_open)
    def test_no_code_response(
        self, mock_file, mock_create_client, config, mock_client, pot_prompt_content
    ):
        """Test handling of response without code."""
        mock_file.return_value.read.return_value = pot_prompt_content
        mock_create_client.return_value = mock_client

        # Response with no code patterns
        no_code_response = self.create_pot_response(
            "This is a simple answer without any programming elements."
        )
        mock_client.generate.return_value = no_code_response

        reasoning = ProgramOfThoughtReasoning(config)
        result = reasoning.execute("Simple question")

        # Should still work but with minimal programming metrics
        assert result.metadata["approach_specific_metrics"]["code_blocks_count"] == 0
        assert (
            result.metadata["approach_specific_metrics"]["contains_variables"] is False
        )
        assert (
            result.metadata["approach_specific_metrics"]["contains_functions"] is False
        )
