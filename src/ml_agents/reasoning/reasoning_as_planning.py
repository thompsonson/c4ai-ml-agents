"""Reasoning-as-Planning approach.

This module implements the Reasoning-as-Planning (RAP) approach,
which treats problem-solving as a strategic planning process with
goal decomposition, action sequencing, and risk assessment.
"""

import re
from pathlib import Path

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class ReasoningAsPlanningReasoning(BaseReasoning):
    """Reasoning-as-Planning approach.

    This class implements the Reasoning-as-Planning methodology,
    which approaches problems through strategic planning principles:

    1. Goal analysis and definition
    2. Problem decomposition into sub-goals
    3. Action sequence development with dependencies
    4. Risk assessment and mitigation planning
    5. Execution strategy formulation

    This approach is particularly effective for complex, multi-step
    problems that benefit from strategic thinking and planning
    methodologies commonly used in project management and AI planning.
    """

    def __init__(self, config) -> None:
        """Initialize the Reasoning-as-Planning approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load Reasoning-as-Planning prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        rap_prompt_path = prompts_dir / "reasoning_as_planning.txt"

        try:
            with open(rap_prompt_path, "r", encoding="utf-8") as f:
                self.rap_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"RAP prompt file not found at {rap_prompt_path}")
            # Fallback RAP prompt
            self.rap_prompt = (
                "Please approach this as a planning problem:\n\n"
                "Question: {question}\n\n"
                "1. Define the goal\n"
                "2. Create a plan\n"
                "3. Consider obstacles\n"
                "4. Execute strategically"
            )

        logger.info("Initialized Reasoning-as-Planning approach")

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Reasoning-as-Planning on the given prompt.

        This method applies the Reasoning-as-Planning methodology to
        approach the problem as a strategic planning exercise with
        goal decomposition and systematic execution planning.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with RAP-enhanced reasoning and metadata
        """
        logger.debug(f"Executing Reasoning-as-Planning on: {prompt[:100]}...")

        # Apply Reasoning-as-Planning prompt template
        rap_enhanced_prompt = self.rap_prompt.format(question=prompt)

        # Get response from API client (auto rate-limited)
        response = self.client.generate(rap_enhanced_prompt)

        # Analyze the response for planning characteristics
        planning_stages = self._count_planning_stages(response.text)
        planning_quality = self._analyze_planning_quality(response.text)

        # Prepare Reasoning-as-Planning specific metadata
        reasoning_data = {
            "reasoning_steps": planning_stages,
            "approach_specific_metrics": {
                "planning_stages_identified": planning_stages,
                "planning_quality_score": planning_quality,
                "contains_goal_analysis": self._has_goal_analysis(response.text),
                "contains_action_sequence": self._has_action_sequence(response.text),
                "contains_risk_assessment": self._has_risk_assessment(response.text),
                "contains_execution_strategy": self._has_execution_strategy(
                    response.text
                ),
                "template_used": "reasoning_as_planning",
                "original_prompt": prompt,
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="reasoning_chain",  # Planning involves step-by-step reasoning
        )

        logger.info(
            f"Completed Reasoning-as-Planning - "
            f"stages: {planning_stages}, tokens: {response.total_tokens}"
        )
        return enhanced_response

    def _count_planning_stages(self, text: str) -> int:
        """Count planning stages in the response.

        This method identifies strategic planning components in the response
        to assess how well the planning methodology was applied.

        Args:
            text: The response text to analyze

        Returns:
            Number of planning stages identified
        """
        # Planning-specific stage indicators
        stage_patterns = [
            r"\b(?:Goal|goal)\s+(?:Analysis|analysis|Definition|definition)",
            r"\b(?:Planning|planning)\s+(?:Phase|phase|Stage|stage)",
            r"\b(?:Action|action)\s+(?:Sequence|sequence|Plan|plan|Steps|steps)",
            r"\b(?:Risk|risk)\s+(?:Assessment|assessment|Analysis|analysis)",
            r"\b(?:Execution|execution)\s+(?:Strategy|strategy|Plan|plan)",
            r"\b(?:Milestone|milestone|Sub-goal|sub-goal|Objective|objective)",
            r"\b(?:Implementation|implementation|Deploy|deploy)",
        ]

        stages_found = 0
        for pattern in stage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                stages_found += 1

        # Also count numbered sections (1., 2., etc.)
        numbered_sections = len(re.findall(r"\d+\.\s+", text))

        return max(stages_found, min(numbered_sections, 6))  # Cap at 6 main stages

    def _analyze_planning_quality(self, text: str) -> float:
        """Analyze the quality of planning in the response.

        Args:
            text: The response text to analyze

        Returns:
            Planning quality score between 0.0 and 1.0
        """
        quality_indicators = {
            "strategic_terms": len(
                re.findall(
                    r"\b(?:strategy|strategic|plan|planning|goal|objective|milestone|roadmap)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "action_words": len(
                re.findall(
                    r"\b(?:implement|execute|deploy|achieve|accomplish|deliver)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "dependency_awareness": len(
                re.findall(
                    r"\b(?:depends on|requires|prerequisite|before|after|once|when)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "risk_awareness": len(
                re.findall(
                    r"\b(?:risk|challenge|obstacle|problem|difficulty|consider|potential)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
        }

        # Calculate quality score (0.0 to 1.0)
        total_indicators = sum(quality_indicators.values())
        text_length_factor = min(
            len(text.split()) / 150, 1.0
        )  # Normalize by text length

        # Quality score based on indicator density and text length
        quality_score = min((total_indicators * 0.08) * text_length_factor, 1.0)
        return round(quality_score, 2)

    def _has_goal_analysis(self, text: str) -> bool:
        """Check if the response contains goal analysis.

        Args:
            text: The response text to analyze

        Returns:
            True if goal analysis is present
        """
        goal_patterns = [
            r"\b(?:goal|objective|aim|target|purpose)\b",
            r"\b(?:what.*achieve|what.*accomplish|what.*need)\b",
            r"\b(?:define|identify|clarify).*(?:goal|objective)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in goal_patterns)

    def _has_action_sequence(self, text: str) -> bool:
        """Check if the response contains action sequencing.

        Args:
            text: The response text to analyze

        Returns:
            True if action sequencing is present
        """
        action_patterns = [
            r"\b(?:step|action|task|activity)\s+\d+",
            r"\b(?:first|second|third|next|then|finally)\b",
            r"\b(?:sequence|order|priority|timeline)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in action_patterns)

    def _has_risk_assessment(self, text: str) -> bool:
        """Check if the response contains risk assessment.

        Args:
            text: The response text to analyze

        Returns:
            True if risk assessment is present
        """
        risk_patterns = [
            r"\b(?:risk|challenge|obstacle|problem|difficulty)\b",
            r"\b(?:potential.*issue|might.*fail|could.*go wrong)\b",
            r"\b(?:mitigation|contingency|backup|alternative)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in risk_patterns)

    def _has_execution_strategy(self, text: str) -> bool:
        """Check if the response contains execution strategy.

        Args:
            text: The response text to analyze

        Returns:
            True if execution strategy is present
        """
        execution_patterns = [
            r"\b(?:implement|execute|deploy|carry out|put into action)\b",
            r"\b(?:strategy|approach|method|technique)\b",
            r"\b(?:how to|way to|process of|steps to)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in execution_patterns)
