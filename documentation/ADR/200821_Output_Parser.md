# ADR-001: Output Parsing Library for ML Agents PoC

**Date:** 2025-08-21
**Status:** Accepted
**Context:** Reasoning research project needs to extract answers from verbose LLM outputs instead of naive string matching.

## Options Evaluated

| Library | Approach | Pros | Cons |
|---------|----------|------|------|
| **Instructor** | Function calling + Pydantic | Multi-provider, retries, proven | Requires function calling support |
| **Pydantic AI** | Function calling | Modern design, streaming | Newer, less examples |
| **Outlines** | Constrained generation | Guaranteed valid output | Local models only, limits creativity |
| **LangChain parsers** | Various | Mature ecosystem | Heavy dependency |
| **Custom regex** | Pattern matching | No dependencies | High maintenance |

## Decision: Test Instructor First

**Why:**

- Works across our target providers (OpenAI, Anthropic, Cohere, OpenRouter)
- Simple integration: `instructor.patch(client)`
- Built-in retries handle API failures
- Pydantic models provide structure for different answer types
- Quick to prototype and evaluate

**Implementation:**

```python
import instructor
from pydantic import BaseModel

class AnswerExtraction(BaseModel):
    final_answer: str
    confidence: float

client = instructor.patch(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    response_model=AnswerExtraction
)
```

**Fallback:** If Instructor doesn't work well across providers or models, fall back to custom regex patterns.

**Success criteria:** Improved accuracy over string matching on sample reasoning outputs from each approach (CoT, ToT, etc.).
