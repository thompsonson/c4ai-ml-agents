"""Tree-of-Thought reasoning approach.

This module implements the Tree-of-Thought (ToT) approach,
which explores multiple reasoning paths, evaluates alternatives,
and synthesizes the best solutions.
"""

import re
from pathlib import Path

from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class TreeOfThoughtReasoning(BaseReasoning):
    """Tree-of-Thought reasoning approach.

    This class implements the Tree-of-Thought methodology,
    which explores multiple reasoning paths systematically:

    1. Problem decomposition into key components
    2. Generation of multiple reasoning branches
    3. Detailed exploration of each branch
    4. Evaluation and comparison of different paths
    5. Selection of the most promising approach(es)
    6. Synthesis of insights from optimal branches

    This approach is particularly effective for complex problems
    with multiple possible solution paths, where exploring
    alternatives leads to better overall solutions.
    """

    def __init__(self, config) -> None:
        """Initialize the Tree-of-Thought approach.

        Args:
            config: Experiment configuration containing model and API settings
        """
        super().__init__(config)

        # Load Tree-of-Thought prompt template
        prompts_dir = Path(__file__).parent / "prompts"
        tot_prompt_path = prompts_dir / "tree_of_thought.txt"

        try:
            with open(tot_prompt_path, "r", encoding="utf-8") as f:
                self.tot_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"ToT prompt file not found at {tot_prompt_path}")
            # Fallback ToT prompt
            self.tot_prompt = (
                "Please explore multiple reasoning paths:\n\n"
                "Question: {question}\n\n"
                "1. Generate different approaches\n"
                "2. Explore each branch\n"
                "3. Evaluate alternatives\n"
                "4. Select best path"
            )

        logger.info("Initialized Tree-of-Thought reasoning approach")

    def execute(self, prompt: str) -> StandardResponse:
        """Execute Tree-of-Thought reasoning on the given prompt.

        This method applies the Tree-of-Thought methodology to
        explore multiple reasoning paths and synthesize the best
        solution through systematic evaluation.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with ToT-enhanced reasoning and metadata
        """
        logger.debug(f"Executing Tree-of-Thought reasoning on: {prompt[:100]}...")

        # Apply Tree-of-Thought prompt template
        tot_enhanced_prompt = self.tot_prompt.format(question=prompt)

        # Get response from API client (auto rate-limited)
        response = self.client.generate(tot_enhanced_prompt)

        # Analyze the response for tree-like reasoning characteristics
        branch_analysis = self._analyze_branches(response.text)
        reasoning_quality = self._analyze_reasoning_quality(response.text)

        # Prepare Tree-of-Thought specific metadata
        reasoning_data = {
            "reasoning_steps": branch_analysis["reasoning_branches"],
            "approach_specific_metrics": {
                "reasoning_branches": branch_analysis["reasoning_branches"],
                "branch_depth": branch_analysis["branch_depth"],
                "evaluation_instances": branch_analysis["evaluations"],
                "reasoning_quality_score": reasoning_quality,
                "contains_problem_decomposition": self._has_problem_decomposition(
                    response.text
                ),
                "contains_reasoning_branches": self._has_reasoning_branches(
                    response.text
                ),
                "contains_branch_exploration": self._has_branch_exploration(
                    response.text
                ),
                "contains_path_evaluation": self._has_path_evaluation(response.text),
                "contains_branch_selection": self._has_branch_selection(response.text),
                "contains_synthesis": self._has_synthesis(response.text),
                "template_used": "tree_of_thought",
                "original_prompt": prompt,
            },
        }

        # Enhance metadata and return
        enhanced_response = self._enhance_metadata(response, reasoning_data)

        # Extract structured answer using output parser
        enhanced_response = self._extract_answer(
            enhanced_response,
            answer_type="reasoning_chain",  # ToT involves exploring multiple reasoning paths
        )

        logger.info(
            f"Completed Tree-of-Thought reasoning - "
            f"branches: {branch_analysis['reasoning_branches']}, "
            f"depth: {branch_analysis['branch_depth']}, "
            f"tokens: {response.total_tokens}"
        )
        return enhanced_response

    def _analyze_branches(self, text: str) -> dict:
        """Analyze the branching characteristics of the response.

        Args:
            text: The response text to analyze

        Returns:
            Dictionary with branch analysis results
        """
        reasoning_branches = self._count_reasoning_branches(text)
        branch_depth = self._analyze_branch_depth(text)
        evaluations = self._count_evaluations(text)

        return {
            "reasoning_branches": reasoning_branches,
            "branch_depth": branch_depth,
            "evaluations": evaluations,
        }

    def _count_reasoning_branches(self, text: str) -> int:
        """Count reasoning branches in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of reasoning branches identified
        """
        # Look for branch indicators
        branch_patterns = [
            r"(?:approach|path|branch|route|method|strategy)\s+\d+",  # "approach 1", "path 2", etc.
            r"(?:alternative|option|possibility|way)\s+\d+",  # "alternative 1", etc.
            r"(?:first|second|third|fourth|fifth|sixth)\s+(?:approach|path|method)",
            r"(?:another|different|alternative)\s+(?:approach|path|way)",
        ]

        branches_found = 0
        for pattern in branch_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            branches_found += len(matches)

        # Also count explicitly numbered alternatives
        numbered_alternatives = len(re.findall(r"(?:^|\n)\s*[A-D]\.\s+", text))
        numbered_approaches = len(re.findall(r"(?:^|\n)\s*\d+\.\s+", text))

        # Take the maximum to avoid double counting
        return max(branches_found, numbered_alternatives, numbered_approaches, 1)

    def _analyze_branch_depth(self, text: str) -> int:
        """Analyze the depth of exploration in branches.

        Args:
            text: The response text to analyze

        Returns:
            Estimated depth of branch exploration
        """
        depth_indicators = [
            r"(?:detailed|in-depth|thorough|comprehensive)\s+(?:exploration|analysis)",
            r"(?:exploring|examining|investigating).*(?:detail|depth)",
            r"(?:pros and cons|advantages and disadvantages|strengths and weaknesses)",
            r"(?:step by step|systematically|methodically)",
        ]

        depth_score = 0
        for pattern in depth_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            depth_score += len(matches)

        # Convert to depth level (1-5 scale)
        if depth_score >= 4:
            return 5
        elif depth_score >= 3:
            return 4
        elif depth_score >= 2:
            return 3
        elif depth_score >= 1:
            return 2
        else:
            return 1

    def _count_evaluations(self, text: str) -> int:
        """Count evaluation instances in the response.

        Args:
            text: The response text to analyze

        Returns:
            Number of evaluation instances found
        """
        evaluation_patterns = [
            r"(?:evaluate|evaluation|assess|assessment|compare|comparison)",
            r"(?:pros|cons|advantages|disadvantages|benefits|drawbacks)",
            r"(?:strengths|weaknesses|better|worse|superior|inferior)",
            r"(?:most promising|best|optimal|preferred|chosen)",
        ]

        evaluation_count = 0
        for pattern in evaluation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            evaluation_count += len(matches)

        return evaluation_count

    def _analyze_reasoning_quality(self, text: str) -> float:
        """Analyze the quality of tree-like reasoning in the response.

        Args:
            text: The response text to analyze

        Returns:
            Reasoning quality score between 0.0 and 1.0
        """
        quality_indicators = {
            "branching_terms": len(
                re.findall(
                    r"\b(?:branch|branches|path|paths|alternative|alternatives|option|options)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "exploration_terms": len(
                re.findall(
                    r"\b(?:explore|exploration|investigate|examination|analysis|detailed)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "evaluation_terms": len(
                re.findall(
                    r"\b(?:evaluate|assessment|compare|pros|cons|strengths|weaknesses)\b",
                    text,
                    re.IGNORECASE,
                )
            ),
            "selection_terms": len(
                re.findall(
                    r"\b(?:select|choose|best|optimal|preferred|promising|synthesis)\b",
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

    def _has_problem_decomposition(self, text: str) -> bool:
        """Check if the response contains problem decomposition.

        Args:
            text: The response text to analyze

        Returns:
            True if problem decomposition is present
        """
        decomposition_patterns = [
            r"\b(?:decomposition|decompose|break down|components|elements)\b",
            r"\b(?:key aspects|main parts|core issues|fundamental elements)\b",
            r"\b(?:divide|separate|analyze|examine).*(?:problem|question|issue)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in decomposition_patterns)

    def _has_reasoning_branches(self, text: str) -> bool:
        """Check if the response contains reasoning branches.

        Args:
            text: The response text to analyze

        Returns:
            True if reasoning branches are present
        """
        branch_patterns = [
            r"\b(?:branch|branches|path|paths|approach|approaches)\b",
            r"\b(?:alternative|alternatives|option|options|method|methods)\b",
            r"(?:different|multiple|various).*(?:way|approach|path|method)",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in branch_patterns)

    def _has_branch_exploration(self, text: str) -> bool:
        """Check if the response contains branch exploration.

        Args:
            text: The response text to analyze

        Returns:
            True if branch exploration is present
        """
        exploration_patterns = [
            r"\b(?:explore|exploration|investigate|examination|develop|developing)\b",
            r"\b(?:in detail|thoroughly|comprehensively|systematically)\b",
            r"(?:each|every).*(?:branch|path|approach|alternative)",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in exploration_patterns)

    def _has_path_evaluation(self, text: str) -> bool:
        """Check if the response contains path evaluation.

        Args:
            text: The response text to analyze

        Returns:
            True if path evaluation is present
        """
        evaluation_patterns = [
            r"\b(?:evaluate|evaluation|assess|assessment|compare|comparison)\b",
            r"\b(?:pros|cons|advantages|disadvantages|strengths|weaknesses)\b",
            r"\b(?:better|worse|superior|inferior|effective|ineffective)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in evaluation_patterns)

    def _has_branch_selection(self, text: str) -> bool:
        """Check if the response contains branch selection.

        Args:
            text: The response text to analyze

        Returns:
            True if branch selection is present
        """
        selection_patterns = [
            r"\b(?:select|selection|choose|chosen|pick|preferred)\b",
            r"\b(?:best|optimal|most promising|ideal|recommended)\b",
            r"\b(?:decision|decide|conclude|final choice)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in selection_patterns)

    def _has_synthesis(self, text: str) -> bool:
        """Check if the response contains synthesis.

        Args:
            text: The response text to analyze

        Returns:
            True if synthesis is present
        """
        synthesis_patterns = [
            r"\b(?:synthesis|synthesize|combine|integration|integrate)\b",
            r"\b(?:bringing together|putting together|merge|unify)\b",
            r"\b(?:insights|learnings|best elements|optimal solution)\b",
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in synthesis_patterns)
