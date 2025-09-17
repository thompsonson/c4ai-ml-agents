"""Chain-of-Verification reasoning approach.

This module implements the Chain-of-Verification (CoVe) reasoning approach,
which enhances accuracy through systematic verification of initial responses.
The approach generates initial answers and then creates verification questions
to double-check the reasoning and conclusions.
"""

from pathlib import Path
from typing import Any

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.logging_config import get_logger
from ml_agents.utils.reasoning_extraction import create_reasoning_prompt_suffix

logger = get_logger(__name__)


class ChainOfVerificationReasoning(BaseReasoning):
    """Chain-of-Verification reasoning approach.

    This class implements the Chain-of-Verification reasoning methodology,
    which improves accuracy by systematically verifying initial responses.

    The approach follows a structured process:
    1. Generate an initial response to the question
    2. Create verification questions about the response
    3. Answer the verification questions
    4. Revise the initial response based on verification results
    5. Provide the final verified answer

    This method is particularly effective for factual questions and scenarios
    where initial responses may contain errors that can be caught through
    systematic verification.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the Chain-of-Verification reasoning approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load CoVe prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        cove_prompt_path = prompts_dir / "chain_of_verification.txt"

        try:
            with open(cove_prompt_path, "r", encoding="utf-8") as f:
                self.cove_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning("CoVe prompt file not found at %s", cove_prompt_path)
            # Fallback CoVe prompt
            self.cove_prompt = (
                "Answer this question step by step, then verify your answer:\n\n"
                "Question: {question}\n\n"
                "Step 1: Provide your initial answer\n"
                "Step 2: Create verification questions\n"
                "Step 3: Answer the verification questions\n"
                "Step 4: Provide your final verified answer"
            )

        logger.info("Initialized Chain-of-Verification reasoning approach")

    def execute(self, prompt: str) -> Any:
        """Execute Chain-of-Verification reasoning on the given prompt.

        Args:
            prompt: The input prompt to reason about

        Returns:
            Structured extraction result from Instructor
        """
        logger.debug("Executing CoVe reasoning on: %s...", prompt[:100])

        # Apply CoVe prompt template with reasoning instructions
        enhanced_prompt = self._prepare_enhanced_prompt(prompt)

        # Use the base class structured extraction method
        extraction = self._execute_with_structured_extraction(enhanced_prompt, prompt)

        logger.info(
            "Completed CoVe reasoning with structured extraction - answer: %s",
            getattr(extraction, "answer_value", "N/A"),
        )
        return extraction

    def _prepare_enhanced_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Prepare enhanced prompt for Chain-of-Verification reasoning approach.

        Args:
            prompt: The original input prompt
            **kwargs: Additional arguments (ignored for CoVe approach)

        Returns:
            Enhanced prompt with Chain-of-Verification-specific instructions
        """
        # Apply CoVe prompt template with reasoning instructions
        reasoning_suffix = create_reasoning_prompt_suffix("chainofverification")
        return self.cove_prompt.format(question=prompt) + reasoning_suffix
