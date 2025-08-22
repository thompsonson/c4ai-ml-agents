"""Chain-of-Thought reasoning approach.

This module implements the Chain-of-Thought (CoT) reasoning approach,
which enhances model responses by encouraging step-by-step thinking
and explicit reasoning chains.
"""

import re
from pathlib import Path

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class ChainOfThoughtReasoning(BaseReasoning):
    """Chain-of-Thought reasoning approach.

    This class implements the Chain-of-Thought reasoning methodology,
    which guides models to break down complex problems into sequential
    reasoning steps, leading to more accurate and interpretable responses.

    The approach uses structured prompting to encourage:
    1. Problem identification and decomposition
    2. Step-by-step logical reasoning
    3. Explicit connection between reasoning steps
    4. Clear final answer derivation

    Research has shown CoT to be particularly effective for mathematical
    reasoning, logical inference, and multi-step problem solving.
    """

    def __init__(self, config) -> None:
        """Initialize the Chain-of-Thought reasoning approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load Chain-of-Thought prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        cot_prompt_path = prompts_dir / "chain_of_thought.txt"

        try:
            with open(cot_prompt_path, "r", encoding="utf-8") as f:
                self.cot_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"CoT prompt file not found at {cot_prompt_path}")
            # Fallback CoT prompt
            self.cot_prompt = (
                "Please think through this step by step:\n\n"
                "Question: {question}\n\n"
                "Let me work through this systematically:"
            )

        logger.info("Initialized Chain-of-Thought reasoning approach")

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Chain-of-Thought reasoning on the given prompt.

        This method applies the Chain-of-Thought methodology to encourage
        the model to break down the problem and reason through it step by step.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with CoT-enhanced reasoning and metadata
        """
        logger.debug(f"Executing Chain-of-Thought reasoning on: {prompt[:100]}...")

        # Apply Chain-of-Thought prompt template
        cot_enhanced_prompt = self.cot_prompt.format(question=prompt)

        # Get response from API client (auto rate-limited)
        response = self.client.generate(cot_enhanced_prompt)

        # Analyze the response for reasoning characteristics
        reasoning_steps = self._count_cot_steps(response.text)
        step_quality = self._analyze_step_quality(response.text)

        # Prepare Chain-of-Thought specific metadata
        reasoning_data = {
            "reasoning_steps": reasoning_steps,
            "approach_specific_metrics": {
                "step_quality_score": step_quality,
                "contains_numbered_steps": self._has_numbered_steps(response.text),
                "contains_logical_connectors": self._has_logical_connectors(
                    response.text
                ),
                "template_used": "chain_of_thought",
                "original_prompt": prompt,
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="reasoning_chain",  # CoT is well-suited for reasoning chains
        )

        logger.info(
            f"Completed Chain-of-Thought reasoning - "
            f"steps: {reasoning_steps}, tokens: {response.total_tokens}"
        )
        return enhanced_response

    def _count_cot_steps(self, text: str) -> int:
        """Count Chain-of-Thought reasoning steps in the response.

        This method provides more accurate step counting specifically
        tailored to Chain-of-Thought response patterns.

        Args:
            text: The response text to analyze

        Returns:
            Number of reasoning steps identified
        """
        # CoT-specific step indicators
        step_patterns = [
            r"\b(?:Step|step)\s+\d+",  # "Step 1", "step 2", etc.
            r"\d+\.\s+",  # "1. ", "2. ", etc.
            r"\b(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)",
            r"\b(?:Initially|Then|Next|After|Finally|Therefore|Thus|Hence)\b",
            r"\b(?:Let me|Let\'s|I need to|I will|I can)\b",
        ]

        total_steps = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_steps += len(matches)

        # Return at least 1 if we found reasoning indicators
        return max(1, total_steps) if total_steps > 0 else 1

    def _analyze_step_quality(self, text: str) -> float:
        """Analyze the quality of reasoning steps in the response.

        Args:
            text: The response text to analyze

        Returns:
            Step quality score between 0.0 and 1.0
        """
        quality_indicators = {
            "logical_connectors": len(
                re.findall(
                    r"\b(?:because|since|therefore|thus|hence|so|consequently)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "structured_thinking": len(
                re.findall(
                    r"\b(?:analyzing|considering|examining|evaluating)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "explicit_reasoning": len(
                re.findall(
                    r"\b(?:this means|this shows|this indicates|this suggests)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
        }

        # Calculate quality score (0.0 to 1.0)
        total_indicators = sum(quality_indicators.values())
        text_length_factor = min(
            len(text.split()) / 100, 1.0
        )  # Normalize by text length

        # Quality score based on indicator density and text length
        quality_score = min((total_indicators * 0.1) * text_length_factor, 1.0)
        return round(quality_score, 2)

    def _has_numbered_steps(self, text: str) -> bool:
        """Check if the response contains numbered steps.

        Args:
            text: The response text to analyze

        Returns:
            True if numbered steps are present
        """
        numbered_pattern = r"\d+\.\s+"
        return len(re.findall(numbered_pattern, text)) >= 2

    def _has_logical_connectors(self, text: str) -> bool:
        """Check if the response contains logical connectors.

        Args:
            text: The response text to analyze

        Returns:
            True if logical connectors are present
        """
        connectors = [
            "therefore",
            "thus",
            "hence",
            "because",
            "since",
            "consequently",
            "as a result",
            "this means",
            "this shows",
        ]

        text_lower = text.lower()
        return any(connector in text_lower for connector in connectors)
