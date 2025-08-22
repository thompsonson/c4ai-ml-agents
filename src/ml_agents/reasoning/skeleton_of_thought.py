"""Skeleton-of-Thought reasoning approach.

This module implements the Skeleton-of-Thought (SoT) approach,
which structures reasoning through hierarchical outline development
and progressive expansion of ideas.
"""

import re
from pathlib import Path

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class SkeletonOfThoughtReasoning(BaseReasoning):
    """Skeleton-of-Thought reasoning approach.

    This class implements the Skeleton-of-Thought methodology,
    which structures complex reasoning through hierarchical thinking:

    1. Create a high-level skeleton outline
    2. Progressive expansion of each outline section
    3. Systematic development of details
    4. Integration of all components
    5. Refinement and polishing

    This approach is particularly effective for complex, multi-faceted
    problems that benefit from structured, hierarchical thinking and
    systematic organization of ideas and arguments.
    """

    def __init__(self, config) -> None:
        """Initialize the Skeleton-of-Thought approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load Skeleton-of-Thought prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        sot_prompt_path = prompts_dir / "skeleton_of_thought.txt"

        try:
            with open(sot_prompt_path, "r", encoding="utf-8") as f:
                self.sot_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"SoT prompt file not found at {sot_prompt_path}")
            # Fallback SoT prompt
            self.sot_prompt = (
                "Please approach this systematically with an outline:\n\n"
                "Question: {question}\n\n"
                "1. Create outline\n"
                "2. Expand each section\n"
                "3. Integrate components\n"
                "4. Final answer"
            )

        logger.info("Initialized Skeleton-of-Thought reasoning approach")

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Skeleton-of-Thought reasoning on the given prompt.

        This method applies the Skeleton-of-Thought methodology to
        structure the problem through hierarchical outline development
        and progressive expansion.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with SoT-enhanced reasoning and metadata
        """
        logger.debug(f"Executing Skeleton-of-Thought reasoning on: {prompt[:100]}...")

        # Apply Skeleton-of-Thought prompt template
        sot_enhanced_prompt = self.sot_prompt.format(question=prompt)

        # Get response from API client (auto rate-limited)
        response = self.client.generate(sot_enhanced_prompt)

        # Analyze the response for structural characteristics
        structure_analysis = self._analyze_structure(response.text)
        outline_quality = self._analyze_outline_quality(response.text)

        # Prepare Skeleton-of-Thought specific metadata
        reasoning_data = {
            "reasoning_steps": structure_analysis["outline_sections"],
            "approach_specific_metrics": {
                "outline_sections": structure_analysis["outline_sections"],
                "hierarchical_levels": structure_analysis["hierarchical_levels"],
                "structure_quality_score": outline_quality,
                "contains_skeleton_outline": self._has_skeleton_outline(response.text),
                "contains_section_expansion": self._has_section_expansion(
                    response.text
                ),
                "contains_progressive_development": self._has_progressive_development(
                    response.text
                ),
                "contains_integration": self._has_integration(response.text),
                "contains_refinement": self._has_refinement(response.text),
                "template_used": "skeleton_of_thought",
                "original_prompt": prompt,
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="textual",  # SoT creates structured textual outlines
        )

        logger.info(
            f"Completed Skeleton-of-Thought reasoning - "
            f"sections: {structure_analysis['outline_sections']}, "
            f"levels: {structure_analysis['hierarchical_levels']}, "
            f"tokens: {response.total_tokens}"
        )
        return enhanced_response

    def _analyze_structure(self, text: str) -> dict:
        """Analyze the structural characteristics of the response.

        Args:
            text: The response text to analyze

        Returns:
            Dictionary with structural analysis results
        """
        outline_sections = self._count_outline_sections(text)
        hierarchical_levels = self._count_hierarchical_levels(text)

        return {
            "outline_sections": outline_sections,
            "hierarchical_levels": hierarchical_levels,
        }

    def _count_outline_sections(self, text: str) -> int:
        """Count outline sections in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of outline sections identified
        """
        # Look for numbered sections, bullet points, and outline patterns
        section_patterns = [
            r"^\s*\d+\.\s+",  # "1. ", "2. ", etc. at line start
            r"^\s*[A-Z]\.\s+",  # "A. ", "B. ", etc. at line start
            r"^\s*[ivx]+\.\s+",  # Roman numerals
            r"^\s*[-*+]\s+",  # Bullet points
            r"(?:Section|section|Part|part)\s+\d+",  # "Section 1", etc.
        ]

        sections_found = 0
        lines = text.split("\n")

        for line in lines:
            for pattern in section_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    sections_found += 1
                    break  # Only count each line once

        return max(sections_found, 1)  # At least 1 if any structure found

    def _count_hierarchical_levels(self, text: str) -> int:
        """Count hierarchical levels in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of hierarchical levels identified
        """
        levels_found = set()
        lines = text.split("\n")

        # Look for different indentation patterns
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue

            # Count leading spaces for indentation levels
            indent = len(line) - len(stripped)
            if indent > 0:
                levels_found.add(indent // 2)  # Assume 2-space indentation

            # Look for numbered hierarchies (1.1, 1.2, 2.1, etc.)
            numbered_hierarchy = re.search(r"(\d+)\.(\d+)", stripped)
            if numbered_hierarchy:
                levels_found.add(2)  # At least 2 levels

            # Look for lettered sub-sections
            lettered = re.search(r"^\s*[a-z]\)", stripped)
            if lettered:
                levels_found.add(2)

        return max(len(levels_found), 1)

    def _analyze_outline_quality(self, text: str) -> float:
        """Analyze the quality of outline structure in the response.

        Args:
            text: The response text to analyze

        Returns:
            Outline quality score between 0.0 and 1.0
        """
        quality_indicators = {
            "structural_terms": len(
                re.findall(
                    r"\b(?:outline|structure|framework|hierarchy|organization|systematic)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "expansion_terms": len(
                re.findall(
                    r"\b(?:expand|expansion|detail|elaborate|develop|build)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "integration_terms": len(
                re.findall(
                    r"\b(?:integrate|integration|connect|combine|synthesis|cohesive)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "hierarchical_terms": len(
                re.findall(
                    r"\b(?:section|subsection|component|element|level|tier)\b",
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

    def _has_skeleton_outline(self, text: str) -> bool:
        """Check if the response contains a skeleton outline.

        Args:
            text: The response text to analyze

        Returns:
            True if skeleton outline is present
        """
        outline_patterns = [
            r"\b(?:skeleton|outline|framework|structure)\b",
            r"\b(?:high-level|overview|main points)\b",
            r"\d+\.\s+\w+.*\n.*\d+\.\s+\w+",  # Multiple numbered items
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in outline_patterns)

    def _has_section_expansion(self, text: str) -> bool:
        """Check if the response contains section expansion.

        Args:
            text: The response text to analyze

        Returns:
            True if section expansion is present
        """
        expansion_patterns = [
            r"\b(?:expand|expansion|detail|elaborate|develop)\b",
            r"\b(?:section|part).*(?:expanded|detailed|elaborated)\b",
            r"(?:going into|diving into|exploring|examining).*(?:detail|depth)",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in expansion_patterns)

    def _has_progressive_development(self, text: str) -> bool:
        """Check if the response contains progressive development.

        Args:
            text: The response text to analyze

        Returns:
            True if progressive development is present
        """
        development_patterns = [
            r"\b(?:progressive|progressively|build|building|develop|developing)\b",
            r"\b(?:step by step|systematically|methodically)\b",
            r"\b(?:building upon|based on|following from)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in development_patterns)

    def _has_integration(self, text: str) -> bool:
        """Check if the response contains integration of components.

        Args:
            text: The response text to analyze

        Returns:
            True if integration is present
        """
        integration_patterns = [
            r"\b(?:integrate|integration|connect|combine|synthesis)\b",
            r"\b(?:bringing together|putting together|cohesive|unified)\b",
            r"\b(?:overall|holistic|comprehensive|complete picture)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in integration_patterns)

    def _has_refinement(self, text: str) -> bool:
        """Check if the response contains refinement.

        Args:
            text: The response text to analyze

        Returns:
            True if refinement is present
        """
        refinement_patterns = [
            r"\b(?:refine|refinement|polish|improve|enhance)\b",
            r"\b(?:final|finalize|conclude|conclusion)\b",
            r"\b(?:comprehensive|complete|thorough)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in refinement_patterns)
