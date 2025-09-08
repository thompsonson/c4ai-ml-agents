"""Base class for all reasoning approaches.

This module provides the abstract base class that all reasoning approaches
must implement, ensuring consistent interfaces and behavior across different
reasoning methodologies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ml_agents.config import ExperimentConfig
from ml_agents.utils.api_clients import StandardResponse, create_api_client
from ml_agents.utils.instructor_clients import InstructorClientManager
from ml_agents.utils.logging_config import get_logger
from ml_agents.utils.reasoning_extraction import REASONING_EXTRACTION_MODELS

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

        # Initialize Instructor client manager with provider-aware configuration
        self.instructor_manager = InstructorClientManager(self.client)

        logger.info(
            f"Initialized {self.approach_name} reasoning approach with provider: {self.client.provider}"
        )

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

    def _execute_with_structured_extraction(
        self, enhanced_prompt: str, original_prompt: str = ""
    ) -> StandardResponse:
        """Execute reasoning with structured extraction using Instructor.

        This method provides a common implementation for structured answer extraction
        that all reasoning approaches can use. It automatically selects the appropriate
        extraction model based on the reasoning approach and handles provider-specific
        Instructor configuration.

        Args:
            enhanced_prompt: The prompt enhanced with reasoning-specific instructions
            original_prompt: The original input prompt (for metadata)

        Returns:
            StandardResponse with structured extraction results

        Raises:
            Exception: If structured extraction fails and no fallback is available
        """
        # Get reasoning-specific extraction model
        approach_key = self.approach_name.lower()
        extraction_model = REASONING_EXTRACTION_MODELS.get(approach_key)

        if not extraction_model:
            logger.warning(
                f"No extraction model found for {approach_key}, using default"
            )
            from ml_agents.utils.reasoning_extraction import NoneReasoningExtraction

            extraction_model = NoneReasoningExtraction

        try:
            # Use Instructor for structured response generation
            extraction = self.instructor_manager.extract_structured_response(
                messages=[{"role": "user", "content": enhanced_prompt}],
                response_model=extraction_model,
                temperature=self.client.temperature,
                max_tokens=self.client.max_tokens,
            )

            # Create StandardResponse with structured data
            response = StandardResponse(
                text=extraction.full_reasoning_text,
                provider=self.client.provider,
                model=self.client.model,
                prompt_tokens=0,  # Will be updated if available from API response
                completion_tokens=0,  # Will be updated if available
                total_tokens=0,  # Will be updated if available
                generation_time=0.0,  # Will be measured by caller
                parameters={
                    "temperature": self.client.temperature,
                    "max_tokens": self.client.max_tokens,
                    "structured_extraction": True,
                },
                extracted_answer=extraction.answer_value,
                metadata={
                    "reasoning_approach": self.approach_name,
                    "reasoning_type": extraction.reasoning_type,
                    "confidence": extraction.confidence,
                    "extraction_method": extraction.extraction_method,
                    "instructor_mode": self.instructor_manager.get_primary_mode(),
                    "original_prompt": original_prompt,
                },
            )

            logger.info(
                f"Structured extraction completed for {self.approach_name}: "
                f"answer='{extraction.answer_value}', confidence={extraction.confidence:.2f}"
            )
            return response

        except Exception as e:
            logger.error(f"Structured extraction failed for {self.approach_name}: {e}")

            # Fallback to original API client generation
            logger.info(f"Falling back to original {self.approach_name} implementation")
            raise e  # Re-raise to let specific reasoning classes handle fallback
