"""Output parser for structured answer extraction using Instructor.

This module provides the OutputParser class that uses Instructor library
with Pydantic models to extract structured answers from LLM outputs.
"""

import time
from typing import Any, Dict, Optional

import instructor
from pydantic import ValidationError

from ml_agents.utils.answer_extraction import (
    BaseAnswerExtraction,
    DefaultExtractionModel,
    get_extraction_model,
)
from ml_agents.utils.api_clients import APIClient
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class ParsingError(Exception):
    """Exception raised when parsing fails."""

    pass


class OutputParser:
    """Parser for extracting structured answers from LLM outputs.

    Uses Instructor library with Pydantic models for structured parsing.

    Args:
        client: API client to use for parsing calls
        use_structured_parsing: Whether to use structured parsing (default: True)
    """

    def __init__(
        self,
        client: APIClient,
        use_structured_parsing: bool = True,
    ) -> None:
        self.client = client
        self.use_structured_parsing = use_structured_parsing

        # Initialize instructor client
        self.instructor_client = None
        if self.use_structured_parsing:
            try:
                self.instructor_client = self._create_instructor_client()
                logger.info(f"Initialized instructor client for {client.provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize instructor client: {e}")
                raise ParsingError(f"Failed to initialize structured parsing: {e}")

    def _create_instructor_client(self) -> Any:
        """Create instructor client based on provider type."""
        try:
            # Get the underlying client for instructor patching
            if hasattr(self.client, "client"):
                # For OpenRouter, Anthropic, etc. that wrap OpenAI/API clients
                base_client = self.client.client
            else:
                # For HuggingFace or other direct clients
                base_client = self.client

            # Patch the client with instructor
            instructor_client = instructor.patch(base_client)
            return instructor_client
        except Exception as e:
            logger.error(f"Failed to patch client with instructor: {e}")
            raise

    def extract_answer(
        self,
        response_text: str,
        answer_type: Optional[str] = None,
        extraction_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured answer from response text.

        Args:
            response_text: Text to extract answer from
            answer_type: Type of answer expected (optional)
            extraction_prompt: Custom extraction prompt (optional)

        Returns:
            Dictionary containing extracted answer and metadata

        Raises:
            ParsingError: If parsing fails and no fallback is available
        """
        start_time = time.time()
        parsing_metadata = {
            "parsing_method": "unknown",
            "parsing_confidence": 0.0,
            "parsing_attempts": 0,
            "extraction_time_ms": 0,
            "errors": [],
        }

        # Use structured parsing
        if not (self.use_structured_parsing and self.instructor_client):
            raise ParsingError("Structured parsing not available")

        result = self._extract_with_instructor(
            response_text, answer_type, extraction_prompt, parsing_metadata
        )

        parsing_metadata["extraction_time_ms"] = int((time.time() - start_time) * 1000)

        if result:
            return {"extraction": result, "metadata": parsing_metadata}
        else:
            raise ParsingError(
                f"Extraction failed. Errors: {parsing_metadata['errors']}"
            )

    def _extract_with_instructor(
        self,
        response_text: str,
        answer_type: Optional[str],
        extraction_prompt: Optional[str],
        metadata: Dict[str, Any],
    ) -> Optional[BaseAnswerExtraction]:
        """Extract answer using Instructor library."""
        from ml_agents.utils.answer_extraction import detect_multiple_tool_calls

        # Detect if response contains multiple tool calls
        has_multiple_calls = detect_multiple_tool_calls(response_text)
        metadata["multiple_tool_calls_detected"] = has_multiple_calls

        # Determine the appropriate model
        if answer_type:
            try:
                model_class = get_extraction_model(
                    answer_type, multiple_responses=has_multiple_calls
                )
            except ValueError:
                logger.warning(f"Unknown answer type: {answer_type}, using default")
                model_class = DefaultExtractionModel
        else:
            # Use multiple response model if tool calls detected
            if has_multiple_calls:
                from ml_agents.utils.answer_extraction import MultipleAnswerExtraction

                model_class = MultipleAnswerExtraction
            else:
                model_class = DefaultExtractionModel

        # Create simple extraction prompt if none provided
        if not extraction_prompt:
            extraction_prompt = (
                f"Extract the final answer from: {response_text[:500]}..."
            )

        # Single extraction attempt
        metadata["parsing_attempts"] = 1
        try:
            logger.debug("Attempting structured extraction")

            # Make the instructor call
            extraction = self.instructor_client.chat.completions.create(
                model=self.client.model,
                messages=[{"role": "user", "content": extraction_prompt}],
                response_model=model_class,
                max_tokens=self.client.max_tokens,
                mode=instructor.Mode.JSON,
            )

            # Handle extraction result
            if has_multiple_calls and hasattr(extraction, "primary_extraction"):
                if extraction.primary_extraction:
                    metadata["parsing_method"] = "instructor_multiple"
                    metadata["parsing_confidence"] = getattr(
                        extraction, "combined_confidence", 0.8
                    )
                    logger.debug("Multiple extraction successful")
                    return extraction.primary_extraction
            else:
                metadata["parsing_method"] = "instructor"
                metadata["parsing_confidence"] = getattr(extraction, "confidence", 0.8)
                logger.debug("Structured extraction successful")
                return extraction

        except ValidationError as e:
            logger.warning(f"Validation error in extraction: {e}")
            metadata["errors"].append(f"Validation error: {str(e)}")
        except Exception as e:
            logger.warning(f"Instructor extraction failed: {e}")
            metadata["errors"].append(f"Instructor error: {str(e)}")

        return None
