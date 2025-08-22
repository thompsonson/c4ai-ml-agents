"""Reflection reasoning approach.

This module implements the Reflection reasoning approach, which enhances
model responses through iterative self-evaluation and refinement.
The approach generates an initial response, reflects on it critically,
and then provides an improved final answer.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

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

    def __init__(self, config) -> None:
        """Initialize the Reflection reasoning approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Configuration for multi-step reflection
        self.max_iterations = getattr(config, "max_reflection_iterations", 2)
        self.reflection_threshold = getattr(config, "reflection_threshold", 0.7)

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

        logger.info(
            f"Initialized Reflection reasoning approach (max_iterations: {self.max_iterations})"
        )

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Reflection reasoning on the given prompt.

        This method applies the Reflection methodology using either single-prompt
        or multi-step approach based on configuration.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with reflection-enhanced reasoning and detailed metadata
        """
        logger.debug(f"Executing Reflection reasoning on: {prompt[:100]}...")

        # Choose execution mode based on configuration
        if getattr(self.config, "multi_step_reflection", False):
            return self._multi_step_reflection(prompt)
        else:
            return self._single_prompt_reflection(prompt)

    def _single_prompt_reflection(self, prompt: str) -> StandardResponse:
        """Execute single-prompt reflection (default mode).

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with reflection-enhanced reasoning
        """
        # Apply Reflection prompt template
        reflection_enhanced_prompt = self.reflection_prompt.format(question=prompt)

        # Execute the reflection process (single call with structured prompt)
        response = self.client.generate(reflection_enhanced_prompt)

        # Parse the reflection structure from the response
        reflection_analysis = self._parse_reflection_structure(response.text)

        # Analyze reflection quality
        reflection_quality = self._analyze_reflection_quality(response.text)
        reflection_steps = self._count_reflection_steps(response.text)

        # Prepare Reflection specific metadata
        reasoning_data = {
            "reasoning_steps": reflection_steps,
            "approach_specific_metrics": {
                "reflection_quality_score": reflection_quality,
                "has_initial_response": reflection_analysis["has_initial"],
                "has_reflection_section": reflection_analysis["has_reflection"],
                "has_refined_response": reflection_analysis["has_refined"],
                "improvement_indicators": reflection_analysis["improvement_count"],
                "self_correction_signals": reflection_analysis["correction_count"],
                "template_used": "reflection",
                "original_prompt": prompt,
                "reflection_structure": reflection_analysis,
            },
            "intermediate_results": {
                "reflection_sections": reflection_analysis["sections"],
                "identified_issues": reflection_analysis["issues"],
                "improvements_made": reflection_analysis["improvements"],
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="reasoning_chain",  # Reflection involves self-evaluation reasoning
        )

        logger.info(
            f"Completed single-prompt Reflection reasoning - "
            f"quality: {reflection_quality}, steps: {reflection_steps}, "
            f"tokens: {response.total_tokens}"
        )
        return enhanced_response

    def _multi_step_reflection(self, prompt: str) -> StandardResponse:
        """Execute multi-step reflection with separate API calls.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with multi-step reflection results
        """
        steps = []
        total_tokens = 0
        total_time = 0.0

        # Step 1: Generate initial response
        initial_prompt = f"Please provide your best answer to this question: {prompt}"
        initial_response = self.client.generate(initial_prompt)

        steps.append(
            {
                "step": "initial",
                "prompt": initial_prompt,
                "response": initial_response.text,
                "tokens": initial_response.total_tokens,
                "time": initial_response.generation_time,
            }
        )

        total_tokens += initial_response.total_tokens
        total_time += initial_response.generation_time

        logger.debug(
            f"Multi-step reflection - Initial response: {len(initial_response.text)} chars"
        )

        # Step 2: Critical reflection
        reflection_prompt = (
            f"Critically examine this answer to the question '{prompt}':\n\n"
            f"Answer: {initial_response.text}\n\n"
            f"What are the potential issues, errors, or areas for improvement? "
            f"Be specific about what could be wrong or incomplete."
        )

        reflection_response = self.client.generate(reflection_prompt)

        steps.append(
            {
                "step": "reflection",
                "prompt": reflection_prompt,
                "response": reflection_response.text,
                "tokens": reflection_response.total_tokens,
                "time": reflection_response.generation_time,
            }
        )

        total_tokens += reflection_response.total_tokens
        total_time += reflection_response.generation_time

        logger.debug(
            f"Multi-step reflection - Reflection: {len(reflection_response.text)} chars"
        )

        # Step 3: Decide if refinement is needed
        final_response = initial_response
        if self._needs_refinement(reflection_response.text):
            # Step 3a: Generate refined response
            refinement_prompt = (
                f"Based on this reflection on your answer:\n\n"
                f"Original Question: {prompt}\n"
                f"Your Answer: {initial_response.text}\n"
                f"Reflection: {reflection_response.text}\n\n"
                f"Please provide an improved, refined answer that addresses the issues identified:"
            )

            refinement_response = self.client.generate(refinement_prompt)

            steps.append(
                {
                    "step": "refinement",
                    "prompt": refinement_prompt,
                    "response": refinement_response.text,
                    "tokens": refinement_response.total_tokens,
                    "time": refinement_response.generation_time,
                }
            )

            total_tokens += refinement_response.total_tokens
            total_time += refinement_response.generation_time
            final_response = refinement_response

            logger.debug(
                f"Multi-step reflection - Refinement: {len(refinement_response.text)} chars"
            )
        else:
            logger.debug(
                "Multi-step reflection - No refinement needed based on threshold"
            )

        # Combine all steps into final response text
        combined_text = self._create_multi_step_response_text(steps)

        # Create comprehensive StandardResponse
        multi_step_response = StandardResponse(
            text=combined_text,
            provider=final_response.provider,
            model=final_response.model,
            prompt_tokens=sum(step["tokens"] for step in steps if "tokens" in step)
            - total_tokens
            + sum(step.get("tokens", 0) for step in steps),  # Approximate prompt tokens
            completion_tokens=total_tokens,
            total_tokens=total_tokens,
            generation_time=total_time,
            parameters=final_response.parameters,
            response_id=final_response.response_id,
            metadata={},
        )

        # Prepare multi-step specific metadata
        reasoning_data = {
            "reasoning_steps": len(steps),
            "approach_specific_metrics": {
                "reflection_quality_score": self._analyze_reflection_quality(
                    combined_text
                ),
                "multi_step_mode": True,
                "steps_completed": len(steps),
                "refinement_applied": len(steps) > 2,
                "total_api_calls": len(steps),
                "template_used": "multi_step_reflection",
                "original_prompt": prompt,
                "refinement_threshold": self.reflection_threshold,
                "max_iterations_allowed": self.max_iterations,
            },
            "intermediate_results": {
                "multi_step_trace": steps,
                "step_breakdown": {
                    "initial_tokens": steps[0].get("tokens", 0),
                    "reflection_tokens": (
                        steps[1].get("tokens", 0) if len(steps) > 1 else 0
                    ),
                    "refinement_tokens": (
                        steps[2].get("tokens", 0) if len(steps) > 2 else 0
                    ),
                },
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(multi_step_response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="reasoning_chain",  # Multi-step reflection involves iterative reasoning
        )

        logger.info(
            f"Completed multi-step Reflection reasoning - "
            f"steps: {len(steps)}, total_tokens: {total_tokens}, "
            f"refinement: {len(steps) > 2}"
        )
        return enhanced_response

    def _needs_refinement(self, reflection_text: str) -> bool:
        """Determine if refinement is needed based on reflection quality.

        Args:
            reflection_text: The reflection analysis text

        Returns:
            True if refinement is recommended
        """
        # Count negative indicators (issues found)
        issue_indicators = [
            "error",
            "mistake",
            "wrong",
            "incorrect",
            "incomplete",
            "missing",
            "overlook",
            "problem",
            "issue",
            "flaw",
            "should",
            "could",
            "might",
            "better",
            "improve",
        ]

        text_lower = reflection_text.lower()
        issue_count = sum(
            1 for indicator in issue_indicators if indicator in text_lower
        )

        # Simple heuristic: if reflection mentions multiple issues, recommend refinement
        issue_density = issue_count / max(len(reflection_text.split()), 1)

        # Apply threshold
        needs_refinement = issue_density > (1 - self.reflection_threshold)

        logger.debug(
            f"Refinement decision - issues: {issue_count}, "
            f"density: {issue_density:.3f}, threshold: {self.reflection_threshold}, "
            f"needs_refinement: {needs_refinement}"
        )

        return needs_refinement

    def _create_multi_step_response_text(self, steps: List[Dict[str, Any]]) -> str:
        """Create combined response text from multi-step process.

        Args:
            steps: List of step dictionaries

        Returns:
            Combined response text
        """
        response_parts = []

        for i, step in enumerate(steps):
            step_name = step["step"].title()
            response_parts.append(f"**{step_name} Response:**")
            response_parts.append(step["response"])
            response_parts.append("")  # Add blank line

        return "\n".join(response_parts).strip()

    def _parse_reflection_structure(self, text: str) -> Dict[str, Any]:
        """Parse the reflection structure from the response text.

        Args:
            text: The response text to analyze

        Returns:
            Dictionary containing parsed reflection structure information
        """
        # Look for reflection structure markers
        has_initial = bool(
            re.search(
                r"\*\*Initial Response\*\*|Initial Response:|First attempt:|My initial answer",
                text,
                re.IGNORECASE,
            )
        )

        has_reflection = bool(
            re.search(
                r"\*\*Reflection\*\*|Reflection:|Let me reflect|Upon reflection",
                text,
                re.IGNORECASE,
            )
        )

        has_refined = bool(
            re.search(
                r"\*\*Refined Response\*\*|Refined Response:|Improved answer:|Better response",
                text,
                re.IGNORECASE,
            )
        )

        # Count improvement indicators
        improvement_patterns = [
            r"\b(?:improve|better|enhance|refine|correct)\b",
            r"\b(?:mistake|error|oversight|miss)\b",
            r"\b(?:should consider|need to|ought to)\b",
        ]

        improvement_count = 0
        for pattern in improvement_patterns:
            improvement_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Count self-correction signals
        correction_patterns = [
            r"\b(?:actually|however|but|wait|no)\b",
            r"\b(?:let me reconsider|on second thought|I realize)\b",
            r"\b(?:correction|revise|adjust)\b",
        ]

        correction_count = 0
        for pattern in correction_patterns:
            correction_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Extract reflection sections
        sections = self._extract_reflection_sections(text)

        # Identify issues and improvements mentioned
        issues = self._extract_issues(text)
        improvements = self._extract_improvements(text)

        return {
            "has_initial": has_initial,
            "has_reflection": has_reflection,
            "has_refined": has_refined,
            "improvement_count": improvement_count,
            "correction_count": correction_count,
            "sections": sections,
            "issues": issues,
            "improvements": improvements,
        }

    def _analyze_reflection_quality(self, text: str) -> float:
        """Analyze the quality of reflection in the response.

        Args:
            text: The response text to analyze

        Returns:
            Reflection quality score between 0.0 and 1.0
        """
        quality_indicators = {
            "self_evaluation": len(
                re.findall(
                    r"\b(?:my response|my answer|I said|I stated|I assumed)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "critical_analysis": len(
                re.findall(
                    r"\b(?:analyze|examine|evaluate|consider|assess)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "improvement_focus": len(
                re.findall(
                    r"\b(?:improve|enhance|better|refine|correct)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "metacognition": len(
                re.findall(
                    r"\b(?:thinking|reasoning|logic|approach|method)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
        }

        # Calculate quality score
        total_indicators = sum(quality_indicators.values())
        text_length_factor = min(len(text.split()) / 200, 1.0)

        # Bonus for structured reflection
        structure_bonus = 0.2 if self._has_structured_reflection(text) else 0.0

        quality_score = min(
            (total_indicators * 0.05) * text_length_factor + structure_bonus, 1.0
        )
        return round(quality_score, 2)

    def _count_reflection_steps(self, text: str) -> int:
        """Count the number of reflection steps in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of reflection steps identified
        """
        # Count major reflection sections
        section_count = 0
        sections = ["initial", "reflection", "refined", "improved", "better"]

        for section in sections:
            if re.search(
                rf"\b{section}\b.*(?:response|answer|attempt)", text, re.IGNORECASE
            ):
                section_count += 1

        # Count reflection actions
        action_patterns = [
            r"\b(?:reflect|consider|reconsider|evaluate|examine)\b",
            r"\b(?:improve|enhance|refine|correct|adjust)\b",
            r"\b(?:realize|notice|see|understand)\b",
        ]

        action_count = 0
        for pattern in action_patterns:
            action_count += len(re.findall(pattern, text, re.IGNORECASE))

        # Total steps is sections + half of actions (to avoid overcounting)
        total_steps = section_count + (action_count // 2)
        return max(2, total_steps)  # Minimum 2 steps for reflection (initial + refined)

    def _has_structured_reflection(self, text: str) -> bool:
        """Check if the response has structured reflection sections.

        Args:
            text: The response text to analyze

        Returns:
            True if structured reflection is present
        """
        structure_markers = [
            r"\*\*(?:Initial|Reflection|Refined)",  # Bold headers
            r"(?:Initial|Reflection|Refined)\s*:",  # Section headers
            r"Step\s+\d+",  # Numbered steps
        ]

        marker_count = 0
        for pattern in structure_markers:
            if re.search(pattern, text, re.IGNORECASE):
                marker_count += 1

        return marker_count >= 2  # At least two structure markers

    def _extract_reflection_sections(self, text: str) -> List[str]:
        """Extract identified reflection sections from the text.

        Args:
            text: The response text to analyze

        Returns:
            List of reflection section names found
        """
        sections = []
        section_patterns = {
            "initial_response": r"\*\*Initial Response\*\*|Initial Response:",
            "reflection": r"\*\*Reflection\*\*|Reflection:",
            "refined_response": r"\*\*Refined Response\*\*|Refined Response:",
            "analysis": r"\*\*Analysis\*\*|Analysis:",
            "improvement": r"\*\*Improvement\*\*|Improvement:",
        }

        for section_name, pattern in section_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                sections.append(section_name)

        return sections

    def _extract_issues(self, text: str) -> List[str]:
        """Extract identified issues from reflection text.

        Args:
            text: The response text to analyze

        Returns:
            List of issues mentioned in reflection
        """
        # This is a simplified extraction - in practice, this could be more sophisticated
        issue_indicators = [
            "error",
            "mistake",
            "oversight",
            "missing",
            "incomplete",
            "wrong",
            "incorrect",
            "overlook",
            "forget",
            "assume",
        ]

        issues = []
        for indicator in issue_indicators:
            if indicator in text.lower():
                issues.append(indicator)

        return list(set(issues))  # Remove duplicates

    def _extract_improvements(self, text: str) -> List[str]:
        """Extract identified improvements from reflection text.

        Args:
            text: The response text to analyze

        Returns:
            List of improvements mentioned in reflection
        """
        # This is a simplified extraction
        improvement_indicators = [
            "better",
            "improve",
            "enhance",
            "refine",
            "clarify",
            "add",
            "include",
            "consider",
            "address",
            "correct",
        ]

        improvements = []
        for indicator in improvement_indicators:
            if indicator in text.lower():
                improvements.append(indicator)

        return list(set(improvements))  # Remove duplicates
