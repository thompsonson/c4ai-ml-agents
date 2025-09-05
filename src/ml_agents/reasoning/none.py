"""None reasoning approach - baseline implementation.

This module provides the baseline "None" reasoning approach that passes
prompts directly to the model without any reasoning enhancement. This
serves as the control group for comparing reasoning effectiveness.
"""

from pathlib import Path

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
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

    def execute(self, prompt: str) -> StandardResponse:
        """Execute baseline reasoning (no enhancement) on the given prompt.

        This method applies minimal processing to the prompt, using only
        the base prompt template without any reasoning-specific enhancements.
        This provides the baseline for measuring reasoning effectiveness.

        Args:
            prompt: The input prompt to process

        Returns:
            StandardResponse with structured reasoning and answer extraction
        """
        logger.debug(f"Executing None reasoning on prompt: {prompt[:100]}...")

        # Apply minimal formatting using base template with reasoning instructions
        reasoning_suffix = create_reasoning_prompt_suffix("none")
        formatted_prompt = self.base_prompt.format(question=prompt) + reasoning_suffix

        try:
            # Use the base class structured extraction method
            response = self._execute_with_structured_extraction(
                formatted_prompt, prompt
            )

            # Add None-specific analysis to metadata
            if response.metadata:
                response.metadata["approach_specific_metrics"] = {
                    "baseline": True,
                    "template_used": "base",
                }
                # Ensure reasoning steps is 0 for baseline
                response.metadata["reasoning_steps"] = 0

            logger.info(
                f"Completed None reasoning with structured extraction - answer: '{response.extracted_answer}'"
            )
            return response

        except Exception as e:
            logger.error(f"Structured extraction failed for None reasoning: {e}")
            # Fallback to original method if Instructor fails
            logger.info("Falling back to original None reasoning implementation")

            # Get unstructured response from API client
            response = self.client.generate(formatted_prompt)

            # Add minimal reasoning metadata
            reasoning_data = {
                "reasoning_steps": 0,
                "approach_specific_metrics": {
                    "baseline": True,
                    "original_prompt": prompt,
                    "template_used": "base",
                    "fallback_used": True,
                    "fallback_reason": str(e),
                },
            }

            # Enhance metadata and extract answer using fallback method
            enhanced_response = self._enhance_metadata(response, reasoning_data)

            # For fallback, try to extract answer using the old output parser method
            try:
                from ml_agents.utils.output_parser import OutputParser

                fallback_parser = OutputParser(
                    client=self.client,
                    use_structured_parsing=False,  # Use regex fallback only
                    fallback_to_regex=True,
                )
                parsing_result = fallback_parser.extract_answer(response.text)
                enhanced_response.extracted_answer = parsing_result[
                    "extraction"
                ].final_answer
                enhanced_response.parsing_metadata = parsing_result["metadata"]
            except Exception as parse_error:
                logger.warning(f"Fallback answer extraction also failed: {parse_error}")
                # Use last sentence as answer
                lines = [
                    line.strip() for line in response.text.split("\n") if line.strip()
                ]
                enhanced_response.extracted_answer = (
                    lines[-1] if lines else response.text[:100]
                )

            logger.info(
                f"Completed None reasoning with fallback - tokens: {response.total_tokens}"
            )
            return enhanced_response
