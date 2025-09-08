"""Reflection reasoning approach.

This module implements the Reflection reasoning approach, which enhances
model responses through iterative self-evaluation and refinement.
The approach generates an initial response, reflects on it critically,
and then provides an improved final answer.
"""

from pathlib import Path

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger
from ml_agents.utils.output_parser import OutputParser
from ml_agents.utils.reasoning_extraction import create_reasoning_prompt_suffix

logger = get_logger(__name__)


class ReflectionReasoning(BaseReasoning):
    """Reflection reasoning approach.

    This class implements the Reflection reasoning methodology, which
    improves response quality through self-evaluation and iterative
    refinement. The approach follows a multi-step process:

    1. Generate an initial response to the question
    2. Critically reflect on the initial response
    3. Identify potential improvements or errors
    4. Generate a refined, improved final response

    This approach is particularly effective for complex problems where
    initial responses may overlook important details or contain errors
    that can be caught through systematic self-evaluation.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the Reflection reasoning approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load Reflection prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        reflection_prompt_path = prompts_dir / "reflection.txt"

        try:
            with open(reflection_prompt_path, "r", encoding="utf-8") as f:
                self.reflection_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(
                f"Reflection prompt file not found at {reflection_prompt_path}"
            )
            # Fallback reflection prompt
            self.reflection_prompt = (
                "Please answer this question, then reflect on your answer and improve it:\n\n"
                "Question: {question}\n\n"
                "First, provide your initial response.\n"
                "Then reflect: What could be wrong or incomplete?\n"
                "Finally, provide your improved answer."
            )

        logger.info("Initialized Reflection reasoning approach")

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Reflection reasoning on the given prompt.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with Reflection reasoning and structured answer extraction
        """
        logger.debug("Executing Reflection reasoning on: %s...", prompt[:100])

        # Apply Reflection prompt template with reasoning instructions
        reasoning_suffix = create_reasoning_prompt_suffix("reflection")
        enhanced_prompt = (
            self.reflection_prompt.format(question=prompt) + reasoning_suffix
        )

        try:
            # Use the base class structured extraction method
            response = self._execute_with_structured_extraction(enhanced_prompt, prompt)

            # Add Reflection-specific analysis to metadata
            if response.metadata:
                response.metadata["approach_specific_metrics"] = {
                    "template_used": "reflection",
                }

            logger.info(
                "Completed Reflection reasoning with structured extraction - answer: %s",
                response.extracted_answer,
            )
            return response

        except Exception as e:
            logger.error("Structured extraction failed for Reflection reasoning: %s", e)
            # Fallback to original method if Instructor fails
            logger.info("Falling back to original Reflection reasoning implementation")

            # Get response from API client
            response = self.client.generate(enhanced_prompt)

            # Basic fallback metadata
            reasoning_data = {
                "approach_specific_metrics": {
                    "template_used": "reflection",
                    "fallback_used": True,
                },
            }

            # Enhance metadata
            enhanced_response = self._enhance_metadata(response, reasoning_data)

            # For fallback, try to extract answer using simple regex
            try:
                fallback_parser = OutputParser(
                    client=self.client,
                    use_structured_parsing=False,
                    fallback_to_regex=True,
                )
                parsing_result = fallback_parser.extract_answer(response.text)
                enhanced_response.extracted_answer = parsing_result[
                    "extraction"
                ].final_answer
                enhanced_response.parsing_metadata = parsing_result["metadata"]
            except Exception as parse_error:
                logger.warning(
                    "Fallback answer extraction also failed: %s", parse_error
                )
                # Use last sentence as answer
                lines = [
                    line.strip() for line in response.text.split("\n") if line.strip()
                ]
                enhanced_response.extracted_answer = (
                    lines[-1] if lines else response.text[:100]
                )

            logger.info(
                "Completed Reflection reasoning with fallback - tokens: %s",
                response.total_tokens,
            )
            return enhanced_response
