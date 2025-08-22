"""Output parser for structured answer extraction using Instructor.

This module provides the OutputParser class that uses Instructor library
with Pydantic models to extract structured answers from LLM outputs,
with graceful fallback to regex patterns when needed.
"""

import re
import time
from typing import Any, Dict, Optional, Type, Union

import instructor
from pydantic import BaseModel, ValidationError

from ml_agents.utils.answer_extraction import (
    BaseAnswerExtraction,
    DefaultExtractionModel,
    get_extraction_model,
)
from ml_agents.utils.api_clients import APIClient, StandardResponse
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class ParsingError(Exception):
    """Exception raised when parsing fails."""

    pass


class OutputParser:
    """Parser for extracting structured answers from LLM outputs.

    Uses Instructor library with Pydantic models for structured parsing,
    with fallback to regex patterns when structured parsing fails.

    Args:
        client: API client to use for parsing calls
        use_structured_parsing: Whether to use structured parsing (default: True)
        fallback_to_regex: Whether to fall back to regex parsing (default: True)
        confidence_threshold: Minimum confidence threshold for accepting results
        max_retries: Maximum number of retry attempts for parsing
    """

    def __init__(
        self,
        client: APIClient,
        use_structured_parsing: bool = True,
        fallback_to_regex: bool = True,
        confidence_threshold: float = 0.7,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.use_structured_parsing = use_structured_parsing
        self.fallback_to_regex = fallback_to_regex
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

        # Initialize instructor client if structured parsing is enabled
        self.instructor_client = None
        if self.use_structured_parsing:
            try:
                self.instructor_client = self._create_instructor_client()
                logger.info(f"Initialized instructor client for {client.provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize instructor client: {e}")
                if not self.fallback_to_regex:
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

        # Try structured parsing first if enabled
        if self.use_structured_parsing and self.instructor_client:
            try:
                result = self._extract_with_instructor(
                    response_text, answer_type, extraction_prompt, parsing_metadata
                )
                if result:
                    parsing_metadata["extraction_time_ms"] = int(
                        (time.time() - start_time) * 1000
                    )
                    return {"extraction": result, "metadata": parsing_metadata}
            except Exception as e:
                logger.warning(f"Structured parsing failed: {e}")
                parsing_metadata["errors"].append(f"Structured parsing: {str(e)}")

        # Fall back to regex parsing if enabled
        if self.fallback_to_regex:
            try:
                result = self._extract_with_regex(response_text, parsing_metadata)
                parsing_metadata["extraction_time_ms"] = int(
                    (time.time() - start_time) * 1000
                )
                return {"extraction": result, "metadata": parsing_metadata}
            except Exception as e:
                logger.warning(f"Regex parsing failed: {e}")
                parsing_metadata["errors"].append(f"Regex parsing: {str(e)}")

        # If all methods fail
        parsing_metadata["extraction_time_ms"] = int((time.time() - start_time) * 1000)
        raise ParsingError(
            f"All parsing methods failed. Errors: {parsing_metadata['errors']}"
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

        # Create extraction prompt
        if not extraction_prompt:
            extraction_prompt = self._create_extraction_prompt(
                response_text, answer_type
            )

        # Try extraction with retries
        for attempt in range(self.max_retries + 1):
            metadata["parsing_attempts"] = attempt + 1
            try:
                logger.debug(
                    f"Attempting structured extraction (attempt {attempt + 1})"
                )

                # Make the instructor call
                extraction = self.instructor_client.chat.completions.create(
                    model=self.client.model,
                    messages=[{"role": "user", "content": extraction_prompt}],
                    response_model=model_class,
                    max_tokens=self.client.max_tokens,
                    temperature=0.1,  # Low temperature for consistent extraction
                )

                # Handle multiple response extraction result
                if has_multiple_calls and hasattr(extraction, "primary_extraction"):
                    # For multiple responses, use the primary extraction
                    if (
                        extraction.primary_extraction
                        and extraction.combined_confidence >= self.confidence_threshold
                    ):
                        metadata["parsing_method"] = "instructor_multiple"
                        metadata["parsing_confidence"] = extraction.combined_confidence
                        metadata["total_extractions"] = len(extraction.extractions)
                        logger.debug(
                            f"Multiple extraction successful with combined confidence {extraction.combined_confidence}"
                        )
                        return extraction.primary_extraction
                    else:
                        logger.debug(
                            f"Multiple extraction confidence {extraction.combined_confidence} below threshold {self.confidence_threshold}"
                        )
                        continue
                else:
                    # Single response handling (existing logic)
                    confidence = getattr(extraction, "confidence", 0.0)
                    if confidence >= self.confidence_threshold:
                        metadata["parsing_method"] = "instructor"
                        metadata["parsing_confidence"] = confidence
                        logger.debug(
                            f"Structured extraction successful with confidence {confidence}"
                        )
                        return extraction
                    else:
                        logger.debug(
                            f"Extraction confidence {confidence} below threshold {self.confidence_threshold}"
                        )
                        continue

            except ValidationError as e:
                logger.warning(f"Validation error in extraction: {e}")
                metadata["errors"].append(f"Validation error: {str(e)}")
            except Exception as e:
                logger.warning(f"Instructor extraction failed: {e}")
                metadata["errors"].append(f"Instructor error: {str(e)}")

                # If it's a rate limit or API error, don't retry immediately
                if "rate limit" in str(e).lower() or "429" in str(e):
                    time.sleep(2**attempt)  # Exponential backoff

        return None

    def _extract_with_regex(
        self, response_text: str, metadata: Dict[str, Any]
    ) -> BaseAnswerExtraction:
        """Extract answer using regex patterns as fallback."""
        logger.debug("Attempting regex extraction")

        # Common regex patterns for answer extraction
        patterns = [
            # Look for "Answer: X" or "Final Answer: X"
            r"(?:final\s+)?answer\s*:\s*(.+?)(?:\n|$)",
            # Look for answers in parentheses or brackets
            r"\(([^)]+)\)$",
            r"\[([^\]]+)\]$",
            # Look for standalone answers at the end
            r"(?:^|\n)([A-Za-z0-9\s\-\.]+)(?:\s*\.?\s*)?$",
            # Look for yes/no answers
            r"\b(yes|no|true|false)\b",
            # Look for multiple choice answers
            r"\b([A-E])\b(?:\)|\.|\s|$)",
        ]

        extracted_text = ""
        confidence = 0.3  # Lower confidence for regex extraction

        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Take the last match (usually the final answer)
                extracted_text = matches[-1].strip()
                confidence = 0.5  # Slightly higher if we found a pattern
                break

        # If no patterns match, use the last line or sentence
        if not extracted_text:
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]
            if lines:
                extracted_text = lines[-1]
                confidence = 0.2  # Very low confidence for last line

        # If still nothing, use truncated response
        if not extracted_text:
            extracted_text = response_text.strip()[:100] + "..."
            confidence = 0.1

        metadata["parsing_method"] = "regex"
        metadata["parsing_confidence"] = confidence

        return DefaultExtractionModel(
            final_answer=extracted_text,
            confidence=confidence,
            extraction_method="regex",
        )

    def _create_extraction_prompt(
        self, response_text: str, answer_type: Optional[str] = None
    ) -> str:
        """Create prompt for answer extraction."""
        base_prompt = f"""
Please extract the final answer from the following response text.

Response to analyze:
{response_text}

Extract the most relevant final answer. If the response contains reasoning steps,
focus on the conclusion. If it contains multiple answers, choose the final one.
Be precise and concise in your extraction.
"""

        if answer_type:
            type_instructions = {
                "multiple_choice": "The answer should be a single letter (A, B, C, D, etc.)",
                "numerical": "Extract the numerical value and any units if present",
                "yes_no": "Extract whether the answer is yes or no",
                "list": "Extract the list items if the answer is a list",
                "textual": "Extract the main textual answer or explanation",
            }

            if answer_type.lower() in type_instructions:
                base_prompt += f"\n\nSpecial instructions: {type_instructions[answer_type.lower()]}"

        return base_prompt

    def is_structured_parsing_available(self) -> bool:
        """Check if structured parsing is available and working."""
        return self.use_structured_parsing and self.instructor_client is not None

    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get parsing statistics and configuration."""
        return {
            "use_structured_parsing": self.use_structured_parsing,
            "fallback_to_regex": self.fallback_to_regex,
            "confidence_threshold": self.confidence_threshold,
            "max_retries": self.max_retries,
            "instructor_client_available": self.instructor_client is not None,
        }
