"""Chain-of-Thought reasoning approach.

This module implements the Chain-of-Thought (CoT) reasoning approach,
which enhances model responses by encouraging step-by-step thinking
and explicit reasoning chains.
"""

from pathlib import Path

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger
from ml_agents.utils.output_parser import OutputParser
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

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Chain-of-Thought reasoning on the given prompt.

        This method applies the Chain-of-Thought methodology to encourage
        the model to break down the problem and reason through it step by step.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with structured CoT reasoning and answer extraction
        """
        logger.debug("Executing Chain-of-Thought reasoning on: %s...", prompt[:100])

        # Apply Chain-of-Thought prompt template with reasoning instructions
        reasoning_suffix = create_reasoning_prompt_suffix("chainofthought")
        cot_enhanced_prompt = self.cot_prompt.format(question=prompt) + reasoning_suffix

        try:
            # Use the base class structured extraction method
            response = self._execute_with_structured_extraction(
                cot_enhanced_prompt, prompt
            )

            # Add Chain-of-Thought specific analysis to metadata
            if response.metadata:
                response.metadata["approach_specific_metrics"] = {
                    "template_used": "chain_of_thought",
                }

            logger.info(
                "Completed Chain-of-Thought reasoning with structured extraction - answer: %s",
                response.extracted_answer,
            )
            return response

        except Exception as e:
            logger.error(
                "Structured extraction failed for Chain-of-Thought reasoning: %s", e
            )
            # Fallback to original method if Instructor fails
            logger.info(
                "Falling back to original Chain-of-Thought reasoning implementation"
            )

            # Get unstructured response from API client
            response = self.client.generate(cot_enhanced_prompt)

            # Prepare Chain-of-Thought specific metadata
            reasoning_data = {
                "approach_specific_metrics": {
                    "template_used": "chain_of_thought",
                    "original_prompt": prompt,
                    "fallback_used": True,
                    "fallback_reason": str(e),
                },
            }

            # Enhance metadata and extract answer using the old method
            enhanced_response = self._enhance_metadata(response, reasoning_data)

            # For fallback, try to extract answer using the old output parser method
            try:

                fallback_parser = OutputParser(
                    client=self.client,
                    use_structured_parsing=False,  # Use regex fallback only
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
                "Completed Chain-of-Thought reasoning with fallback - tokens: %s",
                response.total_tokens,
            )
            return enhanced_response
