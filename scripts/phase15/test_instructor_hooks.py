#!/usr/bin/env python3
"""Test Instructor with hooks to capture raw request/response data."""

import json

import instructor
from openai import OpenAI

from ml_agents.utils.reasoning_extraction import (
    ChainOfThoughtExtraction,
    NoneReasoningExtraction,
)

# Global variables to store captured data
captured_requests = []
captured_responses = []


def capture_request_hook(*args, **kwargs) -> None:
    """Hook to capture raw request data."""
    print("\n🔍 RAW REQUEST CAPTURED:")
    print("Arguments:", args)
    print("Keyword Arguments:")
    print(json.dumps(kwargs, indent=2, default=str))
    captured_requests.append(kwargs)


def capture_response_hook(response) -> None:
    """Hook to capture raw response data."""
    print("\n🔍 RAW RESPONSE CAPTURED:")
    response_dict = (
        response.model_dump() if hasattr(response, "model_dump") else str(response)
    )
    print(json.dumps(response_dict, indent=2, default=str))
    captured_responses.append(response_dict)


def test_with_hooks() -> None:
    """Test both reasoning approaches with Instructor hooks."""

    # Create hooks object
    hooks = instructor.hooks.Hooks()
    hooks.on("completion:kwargs", capture_request_hook)
    hooks.on("completion:response", capture_response_hook)

    client = instructor.from_openai(
        OpenAI(base_url="http://pop-os:8000/v1", api_key="not-needed"),
        hooks=hooks,
        mode=instructor.Mode.JSON,
    )

    test_cases = [
        {
            "name": "None Reasoning",
            "response_model": NoneReasoningExtraction,
            "prompt": "Question: What is 2 + 2?\n\nProvide your complete response and the final answer value separately.",
        },
        {
            "name": "Chain of Thought Reasoning",
            "response_model": ChainOfThoughtExtraction,
            "prompt": "Question: What is 2 + 2?\n\nThink through this step by step. Count your reasoning steps and note if you use numbered steps.\nProvide your complete reasoning and the final answer value separately.",
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TESTING: {test_case['name']}")
        print(f"{'='*60}")

        try:
            extraction = client.chat.completions.create(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                response_model=test_case["response_model"],
                messages=[{"role": "user", "content": test_case["prompt"]}],
                temperature=0.3,
                max_tokens=500,
            )

            print(f"\n✅ INSTRUCTOR EXTRACTED OBJECT:")
            extraction_dict = (
                extraction.model_dump()
                if hasattr(extraction, "model_dump")
                else vars(extraction)
            )
            print(json.dumps(extraction_dict, indent=2, default=str))

            results.append(
                {
                    "test_name": test_case["name"],
                    "request_index": i,
                    "response_index": i,
                    "extracted_object": extraction_dict,
                    "status": "success",
                }
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(
                {"test_name": test_case["name"], "error": str(e), "status": "failed"}
            )

    # Save all captured data to file
    full_data = {
        "captured_requests": captured_requests,
        "captured_responses": captured_responses,
        "extracted_results": results,
        "summary": {
            "total_requests": len(captured_requests),
            "total_responses": len(captured_responses),
            "successful_extractions": len(
                [r for r in results if r["status"] == "success"]
            ),
        },
    }

    with open("instructor_hooks_output.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, default=str)

    print(
        f"\n📄 Complete request/response/extraction data saved to: instructor_hooks_output.json"
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Requests captured: {len(captured_requests)}")
    print(f"Responses captured: {len(captured_responses)}")
    print(
        f"Successful extractions: {len([r for r in results if r['status'] == 'success'])}"
    )


if __name__ == "__main__":
    test_with_hooks()
