"""Base class for all reasoning approaches.

This module provides the abstract base class that all reasoning approaches
must implement, ensuring consistent interfaces and behavior across different
reasoning methodologies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.config import ExperimentConfig
from src.utils.api_clients import StandardResponse, create_api_client
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseReasoning(ABC):
    """Abstract base class for all reasoning approaches.

    This class defines the standard interface that all reasoning approaches
    must implement, providing common functionality and ensuring consistent
    behavior across different reasoning methodologies.

    Args:
        config: Experiment configuration containing model and API settings

    Attributes:
        config: The experiment configuration
        client: The API client for the configured provider
        approach_name: Name of the reasoning approach
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the reasoning approach with configuration.

        Args:
            config: Experiment configuration containing model and API settings
        """
        self.config = config
        self.client = create_api_client(config)
        self.approach_name = self.__class__.__name__.replace("Reasoning", "")

        logger.info(f"Initialized {self.approach_name} reasoning approach")

    @abstractmethod
    def execute(self, prompt: str) -> StandardResponse:
        """Execute the reasoning approach on the given prompt.

        This is the main interface method that all reasoning approaches
        must implement. It should take a prompt and return a StandardResponse
        with reasoning-specific metadata.

        Args:
            prompt: The input prompt to reason about

        Returns:
            StandardResponse with reasoning results and metadata

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement execute method")

    def _enhance_metadata(
        self,
        response: StandardResponse,
        reasoning_data: Optional[Dict[str, Any]] = None,
    ) -> StandardResponse:
        """Enhance the response metadata with reasoning-specific information.

        Args:
            response: The standard response from API client
            reasoning_data: Additional reasoning-specific metadata to add

        Returns:
            StandardResponse with enhanced metadata
        """
        if response.metadata is None:
            response.metadata = {}

        # Add approach identifier
        response.metadata["reasoning_approach"] = self.approach_name

        # Add reasoning-specific data if provided
        if reasoning_data:
            response.metadata.update(reasoning_data)

        logger.debug(f"Enhanced metadata for {self.approach_name}: {reasoning_data}")
        return response

    def _count_reasoning_steps(self, text: str) -> int:
        """Count the number of reasoning steps in the response text.

        This is a helper method that can be overridden by specific approaches
        to provide more accurate step counting based on their output format.

        Args:
            text: The response text to analyze

        Returns:
            Number of reasoning steps identified
        """
        # Default implementation looks for common step indicators
        step_indicators = [
            "Step ",
            "step ",
            "First,",
            "Second,",
            "Third,",
            "Fourth,",
            "Fifth,",
            "1.",
            "2.",
            "3.",
            "4.",
            "5.",
            "6.",
            "7.",
            "8.",
            "9.",
            "10.",
            "Therefore,",
            "Thus,",
            "Hence,",
            "Finally,",
        ]

        step_count = 0
        for indicator in step_indicators:
            step_count += text.count(indicator)

        # Return at least 1 if we found any indicators, otherwise 0
        return max(1, step_count) if step_count > 0 else 0

    def cleanup(self) -> None:
        """Clean up any resources used by the reasoning approach.

        This method should be called when the reasoning approach is no longer
        needed to free up any resources (especially for GPU-based models).
        """
        if hasattr(self.client, "cleanup"):
            self.client.cleanup()
            logger.info(f"Cleaned up resources for {self.approach_name}")
