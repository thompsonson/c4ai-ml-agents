"""Reasoning inference engine for ML Agents experiments.

This module provides the core ReasoningInference class that orchestrates
different reasoning approaches, manages experiments, and tracks costs
and performance metrics across various reasoning methodologies.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning import create_reasoning_approach, get_available_approaches
from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.logging_config import get_logger, log_experiment_start

logger = get_logger(__name__)


@dataclass
class ReasoningResult:
    """Container for reasoning inference results.

    This class encapsulates the results from a reasoning inference operation,
    including the response, performance metrics, and cost tracking information.

    Attributes:
        response: The standardized response from the reasoning approach
        approach_name: Name of the reasoning approach used
        execution_time: Time taken for the reasoning operation (seconds)
        metadata: Additional metadata from the reasoning process
    """

    response: StandardResponse
    approach_name: str
    execution_time: float
    metadata: Dict[str, Any]


class ReasoningInference:
    """Core reasoning inference engine.

    This class orchestrates different reasoning approaches, providing a unified
    interface for executing various reasoning methodologies on prompts. It handles:

    - Reasoning approach instantiation and management
    - Cost tracking and performance monitoring
    - Experiment orchestration and result collection
    - Error handling and fallback mechanisms
    - Batch processing and comparison operations

    The engine integrates with the existing API client infrastructure and
    response standardization.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the reasoning inference engine.

        Args:
            config: Experiment configuration containing model and API settings
        """
        self.config = config
        self.total_cost = 0.0
        self.total_requests = 0
        self.approach_cache: Dict[str, BaseReasoning] = {}

        # Initialize tracking metrics
        self.execution_history: List[ReasoningResult] = []

        # Cost control tracking
        self.reasoning_calls_count = 0
        self.cost_warnings_issued = False

        logger.info("Initialized ReasoningInference engine")
        logger.info(f"Available reasoning approaches: {get_available_approaches()}")

    def run_inference(self, prompt: str, approach_name: str) -> ReasoningResult:
        """Run inference using a specific reasoning approach.

        This is the main interface for executing reasoning on a single prompt
        using the specified approach. It handles approach instantiation,
        execution, cost tracking, and result packaging.

        Args:
            prompt: The input prompt to reason about
            approach_name: Name of the reasoning approach to use

        Returns:
            ReasoningResult containing response and performance metrics

        Raises:
            KeyError: If the reasoning approach is not available
            ValueError: If prompt is empty or invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not approach_name:
            raise ValueError("Reasoning approach name cannot be empty")

        logger.info(f"Running inference with {approach_name} approach")

        # Get or create reasoning approach instance
        reasoning_approach = self._get_or_create_approach(approach_name)

        # Track execution time
        start_time = time.time()

        try:
            # Execute reasoning approach
            response = reasoning_approach.execute(prompt)

            execution_time = time.time() - start_time

            # Create reasoning result
            result = ReasoningResult(
                response=response,
                approach_name=approach_name,
                execution_time=execution_time,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(response.text),
                    "timestamp": datetime.now().isoformat(),
                    "model_used": response.model,
                    "provider_used": response.provider,
                },
            )

            # Update tracking
            self._update_tracking(result)

            logger.info(
                f"Completed {approach_name} inference - " f"time: {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {approach_name} inference: {e}")

            # Create error result with partial information
            error_response = StandardResponse(
                text=f"Error occurred during {approach_name} reasoning: {str(e)}",
                provider=self.config.provider,
                model=self.config.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                generation_time=execution_time,
                parameters=self.config.to_dict(),
                response_id=None,
                metadata={"error": str(e), "approach": approach_name},
            )

            result = ReasoningResult(
                response=error_response,
                approach_name=approach_name,
                execution_time=execution_time,
                metadata={
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "prompt_length": len(prompt),
                },
            )

            self._update_tracking(result)
            raise

    def run_comparison(
        self, prompt: str, approach_names: List[str]
    ) -> Dict[str, ReasoningResult]:
        """Run inference comparison across multiple reasoning approaches.

        This method executes the same prompt across multiple reasoning
        approaches for comparative analysis. It provides a convenient
        interface for benchmarking different approaches.

        Args:
            prompt: The input prompt to reason about
            approach_names: List of reasoning approach names to compare

        Returns:
            Dictionary mapping approach names to their results
        """
        logger.info(f"Running comparison across {len(approach_names)} approaches")

        results = {}
        total_start_time = time.time()

        for approach_name in approach_names:
            try:
                result = self.run_inference(prompt, approach_name)
                results[approach_name] = result

            except Exception as e:
                logger.error(f"Failed to run {approach_name}: {e}")
                # Continue with other approaches
                continue

        total_time = time.time() - total_start_time

        logger.info(
            f"Completed comparison - {len(results)}/{len(approach_names)} successful, "
            f"total time: {total_time:.2f}s"
        )

        return results

    def run_batch_inference(
        self,
        prompts: List[str],
        approach_name: str,
        progress_callback: Optional[callable] = None,
    ) -> List[ReasoningResult]:
        """Run batch inference on multiple prompts using a single approach.

        Args:
            prompts: List of prompts to process
            approach_name: Reasoning approach to use
            progress_callback: Optional callback function for progress updates

        Returns:
            List of ReasoningResult objects for each prompt
        """
        logger.info(
            f"Running batch inference on {len(prompts)} prompts with {approach_name}"
        )

        results = []
        total_start_time = time.time()

        for i, prompt in enumerate(prompts):
            try:
                result = self.run_inference(prompt, approach_name)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(prompts), result)

            except Exception as e:
                logger.error(f"Failed batch item {i+1}: {e}")
                continue

        total_time = time.time() - total_start_time

        logger.info(
            f"Completed batch inference - {len(results)}/{len(prompts)} successful, "
            f"total time: {total_time:.2f}s"
        )

        return results

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of costs and usage across all inferences.

        Returns:
            Dictionary containing cost and usage statistics
        """
        approach_costs = {}
        approach_counts = {}

        for result in self.execution_history:
            approach = result.approach_name
            if approach not in approach_costs:
                approach_costs[approach] = 0.0
                approach_counts[approach] = 0
            approach_counts[approach] += 1

        return {
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
            "approach_costs": approach_costs,
            "approach_counts": approach_counts,
            "average_cost_per_request": self.total_cost / max(1, self.total_requests),
            "execution_count": len(self.execution_history),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all reasoning approaches.

        Returns:
            Dictionary containing performance statistics
        """
        if not self.execution_history:
            return {"message": "No execution history available"}

        approach_times = {}
        approach_tokens = {}

        for result in self.execution_history:
            approach = result.approach_name
            if approach not in approach_times:
                approach_times[approach] = []
                approach_tokens[approach] = []

            approach_times[approach].append(result.execution_time)
            approach_tokens[approach].append(result.response.total_tokens)

        # Calculate averages
        performance_stats = {}
        for approach in approach_times:
            times = approach_times[approach]
            tokens = approach_tokens[approach]

            performance_stats[approach] = {
                "average_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "average_tokens": sum(tokens) / len(tokens),
                "total_executions": len(times),
            }

        return performance_stats

    def _get_or_create_approach(self, approach_name: str) -> BaseReasoning:
        """Get or create a reasoning approach instance.

        Args:
            approach_name: Name of the reasoning approach

        Returns:
            BaseReasoning instance
        """
        if approach_name not in self.approach_cache:
            self.approach_cache[approach_name] = create_reasoning_approach(
                approach_name, self.config
            )

        return self.approach_cache[approach_name]

    def _update_tracking(self, result: ReasoningResult) -> None:
        """Update internal tracking metrics.

        Args:
            result: The reasoning result to track
        """
        self.execution_history.append(result)
        self.total_requests += 1
