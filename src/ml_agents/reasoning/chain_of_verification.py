"""Chain-of-Verification reasoning approach.

This module implements the Chain-of-Verification (CoVe) approach,
which improves response accuracy through systematic verification
questions and iterative refinement.
"""

import re
from pathlib import Path
from typing import Dict, List

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class ChainOfVerificationReasoning(BaseReasoning):
    """Chain-of-Verification reasoning approach.

    This class implements the Chain-of-Verification methodology,
    which enhances answer accuracy through systematic verification:

    1. Generate an initial response to the question
    2. Create specific verification questions
    3. Answer verification questions systematically
    4. Assess accuracy of the initial response
    5. Provide a refined, verified final answer

    The approach supports both single-prompt and multi-step modes:
    - Single-prompt: All verification in one API call
    - Multi-step: Separate calls for initial response and verification

    This method is particularly effective for factual questions,
    complex reasoning tasks, and scenarios where accuracy is critical.
    """

    def __init__(self, config) -> None:
        """Initialize the Chain-of-Verification approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Configuration for multi-step verification
        self.max_iterations = getattr(config, "max_reasoning_calls", 5)
        self.multi_step_verification = getattr(config, "multi_step_verification", False)

        # Load Chain-of-Verification prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        cove_prompt_path = prompts_dir / "chain_of_verification.txt"

        try:
            with open(cove_prompt_path, "r", encoding="utf-8") as f:
                self.cove_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"CoVe prompt file not found at {cove_prompt_path}")
            # Fallback CoVe prompt
            self.cove_prompt = (
                "Please answer this question using verification:\n\n"
                "Question: {question}\n\n"
                "1. Initial answer\n"
                "2. Verification questions\n"
                "3. Check each question\n"
                "4. Final verified answer"
            )

        logger.info(
            f"Initialized Chain-of-Verification approach (multi-step: {self.multi_step_verification})"
        )

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Chain-of-Verification reasoning on the given prompt.

        This method applies the Chain-of-Verification methodology,
        using either single-prompt or multi-step verification based
        on configuration.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with CoVe-enhanced reasoning and metadata
        """
        logger.debug(f"Executing Chain-of-Verification on: {prompt[:100]}...")

        if self.multi_step_verification:
            return self._multi_step_verification(prompt)
        return self._single_prompt_verification(prompt)

    def _single_prompt_verification(self, prompt: str) -> StandardResponse:
        """Execute verification in a single API call.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with single-prompt verification
        """
        logger.debug("Using single-prompt verification mode")

        # Apply Chain-of-Verification prompt template
        cove_enhanced_prompt = self.cove_prompt.format(question=prompt)

        # Get response from API client (auto rate-limited)
        response = self.client.generate(cove_enhanced_prompt)

        # Analyze the response for verification characteristics
        verification_data = self._analyze_verification_response(response.text)

        # Prepare Chain-of-Verification specific metadata
        reasoning_data = {
            "reasoning_steps": verification_data["verification_questions_count"],
            "approach_specific_metrics": {
                "verification_mode": "single_prompt",
                "verification_questions_count": verification_data[
                    "verification_questions_count"
                ],
                "verification_quality_score": verification_data["verification_quality"],
                "contains_initial_response": verification_data["has_initial_response"],
                "contains_verification_questions": verification_data[
                    "has_verification_questions"
                ],
                "contains_accuracy_assessment": verification_data[
                    "has_accuracy_assessment"
                ],
                "contains_refined_response": verification_data["has_refined_response"],
                "template_used": "chain_of_verification",
                "original_prompt": prompt,
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="reasoning_chain",  # CoVe involves verification reasoning chains
        )

        logger.info(
            f"Completed single-prompt Chain-of-Verification - "
            f"questions: {verification_data['verification_questions_count']}, "
            f"tokens: {response.total_tokens}"
        )
        return enhanced_response

    def _multi_step_verification(self, prompt: str) -> StandardResponse:
        """Execute verification with multiple API calls.

        This method performs iterative verification with separate
        API calls for initial response and verification steps.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with multi-step verification
        """
        logger.debug("Using multi-step verification mode")

        # Step 1: Generate initial response
        initial_prompt = (
            f"Please provide your best answer to this question:\n\n{prompt}"
        )
        initial_response = self.client.generate(initial_prompt)

        # Step 2: Generate verification questions
        verification_prompt = (
            f"Given this question and initial answer, generate 3-5 specific verification questions "
            f"to check the accuracy and completeness of the response:\n\n"
            f"Question: {prompt}\n\n"
            f"Initial Answer: {initial_response.text}\n\n"
            f"Verification Questions:"
        )
        verification_response = self.client.generate(verification_prompt)

        # Step 3: Final verification and refinement
        final_prompt = (
            f"Now, answer each verification question and provide a refined final answer:\n\n"
            f"Original Question: {prompt}\n\n"
            f"Initial Answer: {initial_response.text}\n\n"
            f"Verification Questions: {verification_response.text}\n\n"
            f"Please answer each verification question and then provide your refined final answer:"
        )
        final_response = self.client.generate(final_prompt)

        # Combine all responses for analysis
        combined_text = (
            f"Initial Response: {initial_response.text}\n\n"
            f"Verification Questions: {verification_response.text}\n\n"
            f"Final Verification: {final_response.text}"
        )

        # Analyze the combined verification process
        verification_data = self._analyze_verification_response(combined_text)

        # Calculate total token usage
        total_prompt_tokens = (
            initial_response.prompt_tokens
            + verification_response.prompt_tokens
            + final_response.prompt_tokens
        )
        total_completion_tokens = (
            initial_response.completion_tokens
            + verification_response.completion_tokens
            + final_response.completion_tokens
        )
        total_tokens = total_prompt_tokens + total_completion_tokens
        total_time = (
            initial_response.generation_time
            + verification_response.generation_time
            + final_response.generation_time
        )

        # Create combined response using the final response as the base
        combined_response = StandardResponse(
            text=combined_text,
            provider=final_response.provider,
            model=final_response.model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            generation_time=total_time,
            parameters=final_response.parameters,
            response_id=final_response.response_id,
            metadata={
                "multi_step_details": {
                    "initial_response": initial_response.text,
                    "verification_questions": verification_response.text,
                    "final_verification": final_response.text,
                    "step_count": 3,
                }
            },
        )

        # Prepare Chain-of-Verification specific metadata
        reasoning_data = {
            "reasoning_steps": 3,  # Initial, verification, final
            "approach_specific_metrics": {
                "verification_mode": "multi_step",
                "verification_steps": 3,
                "verification_questions_count": verification_data[
                    "verification_questions_count"
                ],
                "verification_quality_score": verification_data["verification_quality"],
                "contains_initial_response": True,
                "contains_verification_questions": True,
                "contains_accuracy_assessment": verification_data[
                    "has_accuracy_assessment"
                ],
                "contains_refined_response": verification_data["has_refined_response"],
                "template_used": "chain_of_verification_multi_step",
                "original_prompt": prompt,
                "total_api_calls": 3,
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(combined_response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="reasoning_chain",  # Multi-step verification with reasoning chains
        )

        logger.info(
            f"Completed multi-step Chain-of-Verification - "
            f"steps: 3, questions: {verification_data['verification_questions_count']}, "
            f"tokens: {total_tokens}"
        )
        return enhanced_response

    def _analyze_verification_response(self, text: str) -> Dict[str, any]:
        """Analyze the verification response for quality metrics.

        Args:
            text: The response text to analyze

        Returns:
            Dictionary with verification analysis results
        """
        verification_questions_count = self._count_verification_questions(text)
        verification_quality = self._analyze_verification_quality(text)

        return {
            "verification_questions_count": verification_questions_count,
            "verification_quality": verification_quality,
            "has_initial_response": self._has_initial_response(text),
            "has_verification_questions": self._has_verification_questions(text),
            "has_accuracy_assessment": self._has_accuracy_assessment(text),
            "has_refined_response": self._has_refined_response(text),
        }

    def _count_verification_questions(self, text: str) -> int:
        """Count verification questions in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of verification questions identified
        """
        # Look for question patterns in verification context
        question_patterns = [
            r"\?\s*(?:\n|$)",  # Lines ending with question marks
            r"(?:Question|question)\s+\d+",  # "Question 1", "question 2", etc.
            r"(?:Verify|verify|Check|check).*\?",  # Verification questions
            r"(?:Is|Are|Does|Did|Can|Will|Would|Should).*\?",  # Question starters
        ]

        total_questions = 0
        for pattern in question_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_questions += len(matches)

        # Remove duplicates by using a more conservative estimate
        return min(total_questions, 10)  # Cap at reasonable number

    def _analyze_verification_quality(self, text: str) -> float:
        """Analyze the quality of verification in the response.

        Args:
            text: The response text to analyze

        Returns:
            Verification quality score between 0.0 and 1.0
        """
        quality_indicators = {
            "verification_terms": len(
                re.findall(
                    r"\b(?:verify|verification|check|validate|confirm|assess|evaluate)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "accuracy_terms": len(
                re.findall(
                    r"\b(?:accurate|accuracy|correct|incorrect|reliable|valid|invalid)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "questioning_terms": len(
                re.findall(
                    r"\b(?:question|doubt|assumption|evidence|support|justify)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "refinement_terms": len(
                re.findall(
                    r"\b(?:refine|refined|improve|improved|better|correction|revise)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
        }

        # Calculate quality score (0.0 to 1.0)
        total_indicators = sum(quality_indicators.values())
        text_length_factor = min(
            len(text.split()) / 200, 1.0
        )  # Normalize by text length

        # Quality score based on indicator density and text length
        quality_score = min((total_indicators * 0.06) * text_length_factor, 1.0)
        return round(quality_score, 2)

    def _has_initial_response(self, text: str) -> bool:
        """Check if the response contains an initial response section.

        Args:
            text: The response text to analyze

        Returns:
            True if initial response is present
        """
        initial_patterns = [
            r"\b(?:initial|initial response|first answer|initial answer)\b",
            r"\b(?:initially|at first|to begin with)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in initial_patterns)

    def _has_verification_questions(self, text: str) -> bool:
        """Check if the response contains verification questions.

        Args:
            text: The response text to analyze

        Returns:
            True if verification questions are present
        """
        verification_patterns = [
            r"\b(?:verification questions?|verify|check)\b.*\?",
            r"\b(?:question|questions?)\b.*(?:verify|check|validate)",
            r"(?:Is|Are|Does|Did|Can|Will|Would|Should).*(?:accurate|correct|valid)\?",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in verification_patterns)

    def _has_accuracy_assessment(self, text: str) -> bool:
        """Check if the response contains accuracy assessment.

        Args:
            text: The response text to analyze

        Returns:
            True if accuracy assessment is present
        """
        assessment_patterns = [
            r"\b(?:accuracy|accurate|assessment|evaluate|evaluation)\b",
            r"\b(?:reliable|reliability|valid|validity|confidence)\b",
            r"\b(?:correct|incorrect|right|wrong|error)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in assessment_patterns)

    def _has_refined_response(self, text: str) -> bool:
        """Check if the response contains a refined answer.

        Args:
            text: The response text to analyze

        Returns:
            True if refined response is present
        """
        refined_patterns = [
            r"\b(?:refined|final|final answer|conclusion)\b",
            r"\b(?:improved|better|corrected|revised)\b",
            r"\b(?:after verification|based on verification)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in refined_patterns)
