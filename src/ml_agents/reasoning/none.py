"""None reasoning approach - baseline implementation.

This module provides the baseline "None" reasoning approach that passes
prompts directly to the model without any reasoning enhancement. This
serves as the control group for comparing reasoning effectiveness.
"""

from pathlib import Path
from typing import Any

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.logging_config import get_logger
from ml_agents.utils.reasoning_extraction import create_reasoning_prompt_suffix

logger = get_logger(__name__)


class NoneReasoning(BaseReasoning):
    """Baseline reasoning approach that applies no reasoning enhancement.

    This class implements a direct pass-through approach where prompts
    are sent to the model without any reasoning-specific modifications.
    It serves as the baseline for measuring the effectiveness of other
    reasoning approaches.

    The approach maintains minimal metadata to ensure fair comparison
    with other reasoning methods while preserving the original prompt
    and response structure.
    """

    def __init__(self, config) -> None:
        """Initialize the None reasoning approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load base prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        base_prompt_path = prompts_dir / "base.txt"

        try:
            with open(base_prompt_path, "r", encoding="utf-8") as f:
                self.base_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Base prompt file not found at {base_prompt_path}")
            self.base_prompt = "Please answer the following question:\n\n{question}"

        logger.info("Initialized None reasoning approach (baseline)")

    def execute(self, prompt: str) -> Any:
        """Execute baseline reasoning (no enhancement) on the given prompt.

        This method applies minimal processing to the prompt, using only
        the base prompt template without any reasoning-specific enhancements.
        This provides the baseline for measuring reasoning effectiveness.

        Args:
            prompt: The input prompt to process

        Returns:
            Structured extraction result from Instructor
        """
        logger.debug(f"Executing None reasoning on prompt: {prompt[:100]}...")

        # Apply minimal formatting using base template with reasoning instructions
        enhanced_prompt = self._prepare_enhanced_prompt(prompt)

        # Use the base class structured extraction method (no fallbacks)
        extraction = self._execute_with_structured_extraction(enhanced_prompt, prompt)

        logger.info(
            f"Completed None reasoning with structured extraction - answer: '{getattr(extraction, 'answer_value', 'N/A')}'"
        )
        return extraction

    def _prepare_enhanced_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Prepare enhanced prompt for None reasoning approach.

        Args:
            prompt: The original input prompt
            **kwargs: Additional arguments (ignored for None approach)

        Returns:
            Enhanced prompt with minimal None-specific formatting
        """
        # Apply minimal formatting using base template with reasoning instructions
        reasoning_suffix = create_reasoning_prompt_suffix("none")
        return self.base_prompt.format(question=prompt) + reasoning_suffix
