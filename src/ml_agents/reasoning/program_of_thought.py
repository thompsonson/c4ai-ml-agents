"""Program-of-Thought reasoning approach.

This module implements the Program-of-Thought (PoT) reasoning approach,
which enhances model responses by encouraging programmatic thinking
and code-based problem solving.
"""

import re
from pathlib import Path

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProgramOfThoughtReasoning(BaseReasoning):
    """Program-of-Thought reasoning approach.

    This class implements the Program-of-Thought reasoning methodology,
    which guides models to solve problems through programmatic thinking,
    code generation, and computational approaches.

    The approach encourages:
    1. Decomposing problems into computational steps
    2. Writing code to solve sub-problems
    3. Using programming logic and data structures
    4. Systematic execution and debugging mindset

    PoT is particularly effective for mathematical problems, data analysis
    tasks, algorithmic challenges, and problems requiring precise computation.
    """

    def __init__(self, config) -> None:
        """Initialize the Program-of-Thought reasoning approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load Program-of-Thought prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        pot_prompt_path = prompts_dir / "program_of_thought.txt"

        try:
            with open(pot_prompt_path, "r", encoding="utf-8") as f:
                self.pot_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"PoT prompt file not found at {pot_prompt_path}")
            # Fallback PoT prompt
            self.pot_prompt = (
                "Let's solve this problem step by step using programming logic:\n\n"
                "Question: {question}\n\n"
                "I'll write code to solve this systematically:"
            )

        logger.info("Initialized Program-of-Thought reasoning approach")

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Program-of-Thought reasoning on the given prompt.

        This method applies the Program-of-Thought methodology to encourage
        the model to approach the problem with computational thinking and
        code-based solutions.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with PoT-enhanced reasoning and metadata
        """
        logger.debug(f"Executing Program-of-Thought reasoning on: {prompt[:100]}...")

        # Apply Program-of-Thought prompt template
        pot_enhanced_prompt = self.pot_prompt.format(question=prompt)

        # Get response from API client (auto rate-limited)
        response = self.client.generate(pot_enhanced_prompt)

        # Analyze the response for programming characteristics
        code_blocks = self._count_code_blocks(response.text)
        programming_quality = self._analyze_programming_quality(response.text)
        computational_steps = self._count_computational_steps(response.text)

        # Prepare Program-of-Thought specific metadata
        reasoning_data = {
            "reasoning_steps": computational_steps,
            "approach_specific_metrics": {
                "code_blocks_count": code_blocks,
                "programming_quality_score": programming_quality,
                "contains_variables": self._has_variables(response.text),
                "contains_functions": self._has_functions(response.text),
                "contains_loops_conditions": self._has_control_structures(
                    response.text
                ),
                "template_used": "program_of_thought",
                "original_prompt": prompt,
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="numerical",  # PoT often produces numerical answers
        )

        logger.info(
            f"Completed Program-of-Thought reasoning - "
            f"code blocks: {code_blocks}, steps: {computational_steps}, "
            f"tokens: {response.total_tokens}"
        )
        return enhanced_response

    def _count_code_blocks(self, text: str) -> int:
        """Count the number of code blocks in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of code blocks identified
        """
        # Look for various code block formats
        patterns = [
            r"```[\w]*\n.*?\n```",  # Markdown code blocks
            r"`[^`\n]+`",  # Inline code
            r"^\s*\w+\s*=",  # Variable assignments
            r"def\s+\w+\(",  # Function definitions
            r"for\s+\w+\s+in\s+",  # For loops
            r"if\s+.*:",  # If statements
        ]

        code_count = 0
        for pattern in patterns:
            matches = re.findall(
                pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE
            )
            code_count += len(matches)

        return code_count

    def _analyze_programming_quality(self, text: str) -> float:
        """Analyze the quality of programming thinking in the response.

        Args:
            text: The response text to analyze

        Returns:
            Programming quality score between 0.0 and 1.0
        """
        quality_indicators = {
            "code_structure": len(
                re.findall(
                    r"\b(?:function|def|class|import|from)\b", text, re.IGNORECASE
                )
            ),
            "computational_thinking": len(
                re.findall(
                    r"\b(?:algorithm|iterate|calculate|compute|process)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "programming_concepts": len(
                re.findall(
                    r"\b(?:variable|array|list|loop|condition|function)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "code_explanation": len(
                re.findall(
                    r"\b(?:this code|the function|this calculates|this returns)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
        }

        # Calculate quality score (0.0 to 1.0)
        total_indicators = sum(quality_indicators.values())
        text_length_factor = min(
            len(text.split()) / 150, 1.0
        )  # Normalize by text length

        # Quality score based on indicator density
        quality_score = min((total_indicators * 0.08) * text_length_factor, 1.0)
        return round(quality_score, 2)

    def _count_computational_steps(self, text: str) -> int:
        """Count computational/programming steps in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of computational steps identified
        """
        step_indicators = [
            r"\b(?:Step|step)\s+\d+",  # Numbered steps
            r"\d+\.\s+",  # Numbered lists
            r"```[\w]*\n",  # Code block starts
            r"\b(?:First|Then|Next|Finally)\b",  # Sequence words
            r"\b(?:Let\'s|Now|We can|To do this)\b",  # Action words
        ]

        total_steps = 0
        for pattern in step_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_steps += len(matches)

        # Also count variable assignments as steps
        assignment_pattern = r"^\s*\w+\s*="
        assignments = re.findall(assignment_pattern, text, re.MULTILINE)
        total_steps += len(assignments)

        return max(1, total_steps) if total_steps > 0 else 1

    def _has_variables(self, text: str) -> bool:
        """Check if the response contains variable usage.

        Args:
            text: The response text to analyze

        Returns:
            True if variables are present
        """
        variable_patterns = [
            r"\b\w+\s*=\s*",  # Variable assignments
            r"\b(?:variable|var)\b",  # Variable mentions
        ]

        for pattern in variable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _has_functions(self, text: str) -> bool:
        """Check if the response contains function definitions or calls.

        Args:
            text: The response text to analyze

        Returns:
            True if functions are present
        """
        function_patterns = [
            r"\bdef\s+\w+\(",  # Python function definitions
            r"\w+\(",  # Function calls
            r"\b(?:function|method)\b",  # Function mentions
        ]

        for pattern in function_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _has_control_structures(self, text: str) -> bool:
        """Check if the response contains control structures (loops, conditions).

        Args:
            text: The response text to analyze

        Returns:
            True if control structures are present
        """
        control_patterns = [
            r"\b(?:if|else|elif|for|while|loop)\b",  # Control keywords
            r"\b(?:iterate|condition|check)\b",  # Control concepts
        ]

        for pattern in control_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
