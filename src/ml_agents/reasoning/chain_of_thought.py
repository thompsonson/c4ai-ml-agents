"""Chain-of-Thought reasoning approach.

This module implements the Chain-of-Thought (CoT) reasoning approach,
which enhances model responses by encouraging step-by-step thinking
and explicit reasoning chains.
"""

from pathlib import Path
from typing import Any

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.logging_config import get_logger
from ml_agents.utils.reasoning_extraction import create_reasoning_prompt_suffix

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

    def __init__(self, config: ExperimentConfig) -> None:
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
            logger.warning("CoT prompt file not found at %s", cot_prompt_path)
            # Fallback CoT prompt
            self.cot_prompt = (
                "Please think through this step by step:\n\n"
                "Question: {question}\n\n"
                "Let me work through this systematically:"
            )

        logger.info("Initialized Chain-of-Thought reasoning approach")

    def execute(self, prompt: str) -> Any:
        """Execute Chain-of-Thought reasoning on the given prompt.

        This method applies the Chain-of-Thought methodology to encourage
        the model to break down the problem and reason through it step by step.

        Args:
            prompt: The input prompt to reason about

        Returns:
            Structured extraction result from Instructor
        """
        logger.debug("Executing Chain-of-Thought reasoning on: %s...", prompt[:100])

        # Apply Chain-of-Thought prompt template with reasoning instructions
        enhanced_prompt = self._prepare_enhanced_prompt(prompt)

        # Use the base class structured extraction method (no fallbacks)
        extraction = self._execute_with_structured_extraction(enhanced_prompt, prompt)

        logger.info(
            "Completed Chain-of-Thought reasoning with structured extraction - answer: %s",
            getattr(extraction, "answer_value", "N/A"),
        )
        return extraction

    def _prepare_enhanced_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Prepare enhanced prompt for Chain-of-Thought reasoning approach.

        Args:
            prompt: The original input prompt
            **kwargs: Additional arguments (ignored for CoT approach)

        Returns:
            Enhanced prompt with Chain-of-Thought-specific instructions
        """
        # Apply Chain-of-Thought prompt template with reasoning instructions
        reasoning_suffix = create_reasoning_prompt_suffix("chainofthought")
        return self.cot_prompt.format(question=prompt) + reasoning_suffix
