"""None reasoning approach - baseline implementation.

This module provides the baseline "None" reasoning approach that passes
prompts directly to the model without any reasoning enhancement. This
serves as the control group for comparing reasoning effectiveness.
"""

from pathlib import Path

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

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

    def execute(self, prompt: str) -> StandardResponse:
        """Execute baseline reasoning (no enhancement) on the given prompt.

        This method applies minimal processing to the prompt, using only
        the base prompt template without any reasoning-specific enhancements.
        This provides the baseline for measuring reasoning effectiveness.

        Args:
            prompt: The input prompt to process

        Returns:
            StandardResponse with minimal reasoning metadata
        """
        logger.debug(f"Executing None reasoning on prompt: {prompt[:100]}...")

        # Apply minimal formatting using base template
        formatted_prompt = self.base_prompt.format(question=prompt)

        # Get response from API client (auto rate-limited)
        response = self.client.generate(formatted_prompt)

        # Add minimal reasoning metadata
        reasoning_data = {
            "reasoning_steps": 0,  # No reasoning steps for baseline
            "approach_specific_metrics": {
                "baseline": True,
                "original_prompt": prompt,
                "template_used": "base",
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response, answer_type="base"  # Simple baseline extraction
        )

        logger.info(f"Completed None reasoning - tokens: {response.total_tokens}")
        return enhanced_response
