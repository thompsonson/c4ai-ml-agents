#!/usr/bin/env python3
"""Test script for structured answer extraction with Instructor."""

import instructor
from openai import OpenAI

from ml_agents.utils.reasoning_extraction import (
    ChainOfThoughtExtraction,
    NoneReasoningExtraction,
)


def test_none_extraction() -> None:
    """Test None reasoning extraction."""
    print("Testing None reasoning extraction...")

    # Create instructor client
    client = instructor.from_openai(
        OpenAI(base_url="http://pop-os:8000/v1", api_key="not-needed")
    )

    prompt = "Question: What is 2 + 10?"

    try:
        extraction = client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            response_model=NoneReasoningExtraction,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            mode=instructor.Mode.JSON,
        )

        print(f"✅ None Extraction Success!")
        print(f"   Answer Value: '{extraction.answer_value}'")
        print(f"   Full Reasoning: '{extraction.full_reasoning_text}'")
        print(f"   Confidence: {extraction.confidence}")
        print(f"   Reasoning Type: {extraction.reasoning_type}")

    except Exception as e:
        print(f"❌ None Extraction Failed: {e}")


def test_cot_extraction() -> None:
    """Test Chain of Thought reasoning extraction."""
    print("\nTesting Chain of Thought extraction...")

    # Create instructor client
    client = instructor.from_openai(
        OpenAI(base_url="http://pop-os:8000/v1", api_key="not-needed")
    )

    prompt = """
    Use the CoT approach and think through this step by step. Count your reasoning steps and note if you use numbered steps.
    Provide your complete reasoning and the final answer value separately.

    Question: What is 2 + 15?
    """

    try:
        extraction = client.chat.completions.create(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            response_model=ChainOfThoughtExtraction,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            mode=instructor.Mode.JSON,
        )

        print(f"✅ Chain of Thought Extraction Success!")
        print(f"   Answer Value: '{extraction.answer_value}'")
        print(f"   Full Reasoning: '{extraction.full_reasoning_text}'")
        print(f"   Step Count: {extraction.step_count}")
        print(f"   Contains Numbered Steps: {extraction.contains_numbered_steps}")
        print(f"   Confidence: {extraction.confidence}")
        print(f"   Reasoning Type: {extraction.reasoning_type}")

    except Exception as e:
        print(f"❌ Chain of Thought Extraction Failed: {e}")


if __name__ == "__main__":
    test_none_extraction()
    test_cot_extraction()
