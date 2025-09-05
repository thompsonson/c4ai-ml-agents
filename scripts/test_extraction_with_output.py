#!/usr/bin/env python3
"""Test structured extraction and save detailed results to files."""

import csv
import json
from datetime import datetime

import instructor
from openai import OpenAI

from ml_agents.utils.reasoning_extraction import (
    ChainOfThoughtExtraction,
    NoneReasoningExtraction,
)


def test_and_save_results():
    """Test both reasoning approaches and save detailed results."""

    client = instructor.from_openai(
        OpenAI(base_url="http://pop-os:8000/v1", api_key="not-needed")
    )

    test_cases = [
        {
            "reasoning_type": "none",
            "response_model": NoneReasoningExtraction,
            "prompt": """
Question: What is 2 + 2?

Provide your complete response and the final answer value separately.
""",
        },
        {
            "reasoning_type": "chain_of_thought",
            "response_model": ChainOfThoughtExtraction,
            "prompt": """
Question: What is 2 + 2?

Think through this step by step. Count your reasoning steps and note if you use numbered steps.
Provide your complete reasoning and the final answer value separately.
""",
        },
    ]

    results = []
    timestamp = datetime.now().isoformat()

    for test_case in test_cases:
        print(f"Testing {test_case['reasoning_type']} reasoning...")

        # Debug: Show request details
        messages = [{"role": "user", "content": test_case["prompt"]}]
        print(f"\n🔍 DEBUG - REQUEST:")
        print(f"   Model: Qwen/Qwen2.5-1.5B-Instruct")
        print(f"   Response Model: {test_case['response_model'].__name__}")
        print(f"   Messages: {json.dumps(messages, indent=4)}")

        try:
            # Make the API call with instructor
            extraction = client.chat.completions.create(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                response_model=test_case["response_model"],
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )

            # Debug: Show raw response (what Instructor extracted)
            print(f"\n🔍 DEBUG - INSTRUCTOR EXTRACTED RESPONSE:")
            extraction_dict = (
                extraction.dict() if hasattr(extraction, "dict") else vars(extraction)
            )
            print(
                f"   Raw Extraction Object: {json.dumps(extraction_dict, indent=4, default=str)}"
            )

            # Collect all the data
            result = {
                "timestamp": timestamp,
                "reasoning_type": test_case["reasoning_type"],
                "input_prompt": test_case["prompt"].strip(),
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "temperature": 0.3,
                "max_tokens": 500,
                # DEBUG: Raw request/response
                "debug_request_messages": messages,
                "debug_response_model": test_case["response_model"].__name__,
                "debug_extracted_object": extraction_dict,
                # EXTRACTED STRUCTURED DATA
                "extracted_answer": extraction.answer_value,  # CLEAN ANSWER
                "full_reasoning_text": extraction.full_reasoning_text,  # COMPLETE REASONING
                "confidence": extraction.confidence,
                "extraction_method": extraction.extraction_method,
                "reasoning_type_field": extraction.reasoning_type,
                # Additional fields for Chain of Thought
                "step_count": getattr(extraction, "step_count", None),
                "contains_numbered_steps": getattr(
                    extraction, "contains_numbered_steps", None
                ),
                # Status
                "extraction_status": "success",
            }

            print(f"\n✅ {test_case['reasoning_type']} extraction successful")
            print(f"   Extracted Answer: '{extraction.answer_value}'")
            print(f"   Full Reasoning: '{extraction.full_reasoning_text}'")

        except Exception as e:
            print(f"\n🔍 DEBUG - ERROR DETAILS:")
            print(f"   Exception Type: {type(e).__name__}")
            print(f"   Exception Message: {str(e)}")

            result = {
                "timestamp": timestamp,
                "reasoning_type": test_case["reasoning_type"],
                "input_prompt": test_case["prompt"].strip(),
                "debug_request_messages": messages,
                "debug_response_model": test_case["response_model"].__name__,
                "extraction_status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            print(f"❌ {test_case['reasoning_type']} extraction failed: {e}")

        results.append(result)
        print("\n" + "-" * 60)

    # Save results to JSON
    json_file = "extraction_test_results_debug.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Detailed results with debug info saved to: {json_file}")

    # Save results to CSV (excluding complex debug objects)
    csv_file = "extraction_test_results_debug.csv"
    if results:
        # Create CSV-friendly version (exclude complex debug objects)
        csv_results = []
        for result in results:
            csv_result = result.copy()
            # Convert complex objects to strings for CSV
            if "debug_request_messages" in csv_result:
                csv_result["debug_request_messages"] = json.dumps(
                    csv_result["debug_request_messages"]
                )
            if "debug_extracted_object" in csv_result:
                csv_result["debug_extracted_object"] = json.dumps(
                    csv_result["debug_extracted_object"], default=str
                )
            csv_results.append(csv_result)

        fieldnames = set()
        for result in csv_results:
            fieldnames.update(result.keys())
        fieldnames = sorted(list(fieldnames))

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results)
    print(f"📄 CSV results saved to: {csv_file}")

    return results


if __name__ == "__main__":
    test_results = test_and_save_results()

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPARISON SUMMARY")
    print("=" * 60)

    for result in test_results:
        if result.get("extraction_status") == "success":
            print(f"\n{result['reasoning_type'].upper()} REASONING:")
            print(f"  Input: 'What is 2 + 2?'")
            print(
                f"  Extracted Answer: '{result['extracted_answer']}'"
            )  # Should be just "4"
            print(f"  Full Reasoning: '{result['full_reasoning_text']}'")
            if result.get("step_count"):
                print(f"  Steps: {result['step_count']}")
