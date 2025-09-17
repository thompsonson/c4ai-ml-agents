"""Base class for all reasoning approaches.

This module provides the abstract base class that all reasoning approaches
must implement, ensuring consistent interfaces and behavior across different
reasoning methodologies.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ml_agents.config import ExperimentConfig
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
        instructor_manager: The Instructor client manager for structured output
        approach_name: Name of the reasoning approach
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the reasoning approach with configuration.

        Args:
            config: Experiment configuration containing model and API settings
        """
        self.config = config
        self.approach_name = self.__class__.__name__.replace("Reasoning", "")

        # Initialize Instructor client manager with direct config integration
        self.instructor_manager = InstructorClientManager(config)

        logger.info(
            f"Initialized {self.approach_name} reasoning approach with provider: {config.provider}"
        )

    @abstractmethod
    def execute(self, prompt: str) -> Any:
        """Execute the reasoning approach on the given prompt.

        This is the main interface method that all reasoning approaches
        must implement. It should take a prompt and return a StandardResponse
        with reasoning-specific metadata.

        Args:
            prompt: The input prompt to reason about

        Returns:
            Structured reasoning result from Instructor

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement execute method")

    def _enhance_extraction_metadata(self, extraction: Any) -> Dict[str, Any]:
        """Create metadata dictionary for structured extraction results.

        Args:
            extraction: The structured extraction result from Instructor

        Returns:
            Dictionary with reasoning-specific metadata
        """
        metadata = {
            "reasoning_approach": self.approach_name,
            "provider": self.config.provider,
            "model": self.config.model,
            "instructor_mode": self.instructor_manager.get_primary_mode(),
        }

        # Add extraction-specific metadata if available
        if hasattr(extraction, "confidence"):
            metadata["confidence"] = extraction.confidence
        if hasattr(extraction, "reasoning_type"):
            metadata["reasoning_type"] = extraction.reasoning_type
        if hasattr(extraction, "extraction_method"):
            metadata["extraction_method"] = extraction.extraction_method

        logger.debug(f"Created metadata for {self.approach_name}: {metadata}")
        return metadata

    def _execute_with_structured_extraction(
        self, enhanced_prompt: str, original_prompt: str = ""
    ) -> Any:
        """Execute reasoning with structured extraction using Instructor.

        This method provides a common implementation for structured answer extraction
        that all reasoning approaches can use. It automatically selects the appropriate
        extraction model based on the reasoning approach and handles provider-specific
        Instructor configuration.

        Args:
            enhanced_prompt: The prompt enhanced with reasoning-specific instructions
            original_prompt: The original input prompt (for metadata)

        Returns:
            Structured extraction result from Instructor

        Raises:
            Exception: If structured extraction fails
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
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            logger.info(
                f"Structured extraction completed for {self.approach_name}: "
                f"answer='{getattr(extraction, 'answer_value', 'N/A')}', confidence={getattr(extraction, 'confidence', 0.0):.2f}"
            )
            return extraction

        except Exception as e:
            logger.error(f"Structured extraction failed for {self.approach_name}: {e}")
            raise e

    @abstractmethod
    def _prepare_enhanced_prompt(self, prompt: str, **kwargs: Any) -> str:
        """Prepare single prompt with reasoning-specific enhancements.

        This method must be implemented by subclasses to apply their specific
        prompt enhancement logic.

        Args:
            prompt: The original input prompt
            **kwargs: Additional arguments for prompt enhancement

        Returns:
            Enhanced prompt with reasoning-specific instructions
        """
        raise NotImplementedError(
            "Subclasses must implement _prepare_enhanced_prompt method"
        )

    async def execute_concurrent(
        self, prompts: List[str], concurrency_limit: int = 5, **kwargs: Any
    ) -> List[Any]:
        """Execute reasoning on multiple prompts concurrently via Instructor.

        This method provides concurrent execution of reasoning approaches using
        the InstructorClientManager async capabilities. It maintains the same
        structured extraction functionality as the single execute() method.

        Args:
            prompts: List of input prompts to reason about
            concurrency_limit: Maximum number of concurrent executions
            **kwargs: Additional arguments for reasoning execution

        Returns:
            List of structured reasoning results from Instructor

        Raises:
            Exception: If concurrent structured extraction fails
        """
        import time

        concurrent_start_time = time.time()
        logger.info(
            f"🚀 Starting concurrent {self.approach_name} reasoning on {len(prompts)} prompts with {concurrency_limit} concurrent workers (provider: {self.config.provider})"
        )

        # Prepare enhanced prompts for all inputs using subclass-specific logic
        enhanced_prompts = [
            self._prepare_enhanced_prompt(prompt, **kwargs) for prompt in prompts
        ]

        # Debug logging to check if enhanced_prompts contains None values
        none_count = sum(1 for ep in enhanced_prompts if ep is None)
        if none_count > 0:
            logger.error(
                f"❌ Found {none_count} None enhanced prompts out of {len(enhanced_prompts)} total!"
            )
            logger.debug(f"First few enhanced prompts: {enhanced_prompts[:5]}")

        # Get reasoning-specific extraction model
        approach_key = self.approach_name.lower()
        extraction_model = REASONING_EXTRACTION_MODELS.get(approach_key)

        if not extraction_model:
            logger.warning(
                f"No extraction model found for {approach_key}, using default"
            )
            from ml_agents.utils.reasoning_extraction import NoneReasoningExtraction

            extraction_model = NoneReasoningExtraction

        # Prepare messages for Instructor, filtering out None prompts
        messages_list = []
        for i, enhanced_prompt in enumerate(enhanced_prompts):
            if enhanced_prompt is not None:
                messages_list.append([{"role": "user", "content": enhanced_prompt}])
            else:
                logger.error(f"❌ Skipping prompt {i} due to None enhanced_prompt")
                messages_list.append(None)  # Maintain index alignment

        # Execute concurrent structured extraction directly
        return await self._execute_concurrent_async(
            messages_list,
            extraction_model,
            concurrency_limit,
            concurrent_start_time,
            **kwargs,
        )

    async def _execute_concurrent_async(
        self,
        messages_list,
        extraction_model,
        concurrency_limit,
        concurrent_start_time,
        **kwargs,
    ):
        """Internal async method for concurrent execution."""
        import time

        try:
            # Execute concurrent structured extraction
            extractions = await self.instructor_manager.extract_concurrent_responses(
                messages_list=messages_list,
                response_model=extraction_model,
                concurrency_limit=concurrency_limit,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            )

            # Return the raw extractions - let callers handle conversion if needed
            successful_count = sum(
                1
                for extraction in extractions
                if extraction is not None and not isinstance(extraction, Exception)
            )

            concurrent_end_time = time.time()
            total_duration = concurrent_end_time - concurrent_start_time
            logger.info(
                f"🏁 Concurrent {self.approach_name} reasoning completed in {total_duration:.2f}s: "
                f"{successful_count}/{len(extractions)} successful extractions (avg: {total_duration/len(extractions):.2f}s per request)"
            )

            return extractions

        except Exception as e:
            logger.error(f"Concurrent {self.approach_name} reasoning failed: {e}")
            raise e
