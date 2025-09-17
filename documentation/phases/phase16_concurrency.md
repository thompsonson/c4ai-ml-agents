# Concurrent API Calls Specification (Revised)

## Objective

Add concurrent API call support to reasoning approaches for increased throughput with vLLM endpoints.

**Note**: This is concurrency (multiple simultaneous single API calls), not batching (multiple prompts in one API call).

**Critical Update**: All reasoning approaches use structured extraction via InstructorClientManager, not direct API clients.

**Instructor Async Documentation**: https://python.useinstructor.com/blog/2023/11/13/learn-async/

## Current Architecture (Actual)

```
ExperimentRunner → ReasoningApproach.execute(prompt) →
BaseReasoning._execute_with_structured_extraction() →
InstructorClientManager.extract_structured_response() →
instructor.chat.completions.create() → StandardResponse
```

## Proposed Architecture (Corrected)

```
ExperimentRunner → ReasoningApproach.execute_concurrent(prompts) →
InstructorClientManager.extract_structured_response_async() →
[await instructor.chat.completions.create() for prompt in prompts] → List[StandardResponse]
```

## Core Changes Required

### 1. InstructorClientManager Layer (`utils/instructor_clients.py`)

Add async structured extraction method:

```python
async def extract_structured_response_async(
    self,
    messages: List[Dict],
    response_model: Type[BaseModel],
    **kwargs
) -> BaseModel:
    """Async version of structured response extraction - Instructor only."""
    # Pure Instructor async implementation - NO FALLBACKS
    response = await self.client.chat.completions.create(
        model=self.model_name,
        response_model=response_model,
        messages=messages,
        **kwargs
    )
    return response

async def extract_concurrent_responses(
    self,
    messages_list: List[List[Dict]],
    response_model: Type[BaseModel],
    concurrency_limit: int = 10,
    **kwargs
) -> List[BaseModel]:
    """Extract multiple structured responses concurrently."""
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def bounded_extract(messages):
        async with semaphore:
            return await self.extract_structured_response_async(
                messages, response_model, **kwargs
            )

    tasks = [bounded_extract(msgs) for msgs in messages_list]
    return await asyncio.gather(*tasks)
```

### 2. Reasoning Base Class (`reasoning/base.py`)

Add concurrent execution method:

```python
async def execute_concurrent(self, prompts: List[str]) -> List[StandardResponse]:
    """Execute reasoning on multiple prompts concurrently via Instructor."""
    # Prepare enhanced prompts for all inputs
    enhanced_prompts = [self._prepare_enhanced_prompt(prompt) for prompt in prompts]

    # Get reasoning-specific extraction model
    approach_key = self.approach_name.lower()
    extraction_model = REASONING_EXTRACTION_MODELS.get(approach_key)

    # Prepare messages for Instructor
    messages_list = [
        [{"role": "user", "content": enhanced_prompt}]
        for enhanced_prompt in enhanced_prompts
    ]

    # Execute concurrent structured extraction
    extractions = await self.instructor_manager.extract_concurrent_responses(
        messages_list=messages_list,
        response_model=extraction_model,
        concurrency_limit=10,
        temperature=self.client.temperature,
        max_tokens=self.client.max_tokens,
    )

    # Convert to StandardResponse format
    responses = []
    for i, extraction in enumerate(extractions):
        response = StandardResponse(
            text=extraction.full_reasoning_text,
            provider=self.client.provider,
            model=self.client.model,
            extracted_answer=extraction.answer_value,
            metadata={
                "reasoning_approach": self.approach_name,
                "reasoning_type": extraction.reasoning_type,
                "confidence": extraction.confidence,
                "original_prompt": prompts[i],
                "concurrent_execution": True,
            }
        )
        responses.append(response)

    return responses

def _prepare_enhanced_prompt(self, prompt: str) -> str:
    """Prepare single prompt with reasoning-specific enhancements."""
    # Extract from existing execute() method logic
```

### 3. Experiment Runner Integration

Add concurrent processing option:

```python
async def run_reasoning_concurrent(
    self,
    prompts: List[str],
    reasoning_approach: BaseReasoning,
    concurrency_limit: int = 10
) -> List[StandardResponse]:
    """Process prompts with concurrent structured extraction."""
    return await reasoning_approach.execute_concurrent(prompts)
```

## Implementation Notes

### Dependencies

- Verify Instructor library async support: `await client.chat.completions.create()`
- Provider-specific validation for Local-OpenAI (vLLM endpoint)
- Maintain existing primary/fallback mode logic from InstructorClientManager

### Configuration

- Add `--concurrent` flag to enable concurrent processing
- Add `--concurrency-limit` CLI parameter (default: 10)
- Focus on Local-OpenAI provider initially

### Structured Extraction Preservation

- **Critical**: Maintain all Phase 15 structured extraction capabilities
- All reasoning approaches continue using same extraction models
- Primary/fallback mode logic preserved for async operations
- Same response format and metadata structure

### Backward Compatibility

- Keep existing `execute(prompt)` method unchanged
- New `execute_concurrent()` method is additive
- Existing experiments continue working without changes

## Success Criteria

1. 10 concurrent structured extraction calls execute simultaneously
2. Full Phase 15 structured extraction capabilities preserved
3. 2-5x throughput improvement for vLLM endpoints
4. Configurable concurrency limit for throughput tuning
5. No changes to response format or metadata structure
6. Existing synchronous interface remains functional
7. Failed prompts don't block successful ones (partial results)
