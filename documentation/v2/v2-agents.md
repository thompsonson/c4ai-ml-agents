# ML Agents v2 Reasoning Agents Specification

**Version:** 1.1
**Date:** 2025-09-17
**Purpose:** Define reasoning agent implementations and behavioral specifications

## Overview

Reasoning agents are domain services that apply different reasoning approaches to individual questions. Each agent encapsulates a specific reasoning methodology while maintaining consistent interfaces for evaluation orchestration.

## Agent Interface

All reasoning agents implement the `ReasoningAgentService` interface:

```python
class ReasoningAgentService(Protocol):
    async def process_question(self, question: Question, config: AgentConfig) -> Union[Answer, FailureReason]
    def get_agent_type(self) -> str
    def validate_config(self, config: AgentConfig) -> ValidationResult
    def get_required_parameters(self) -> List[str]
```

## Core Agent Implementations

### 1. None Agent (Direct Response)

**Purpose**: Baseline approach where the model answers questions directly without explicit reasoning steps.

**Behavioral Pattern**:

- Prompts model for direct, concise answers
- No reasoning steps required or captured
- Serves as comparison baseline for reasoning approaches

**Prompt Template Concept**:

```
Answer the following question directly and concisely:
Question: {question.text}
Provide only the final answer without showing your work.
```

**Output Characteristics**:

- Empty reasoning trace (reasoning_text = "")
- Clean extracted answer
- Minimal token usage
- Fast response times

### 2. Chain of Thought Agent

**Purpose**: Explicit step-by-step reasoning before providing the final answer.

**Behavioral Pattern**:

- Prompts model to show reasoning process explicitly
- Captures reasoning steps in trace
- Separates reasoning from final answer

**Prompt Template Concept**:

```
Answer the following question by thinking through it step by step:
Question: {question.text}
Show your reasoning process clearly, then provide your final answer.
```

**Output Characteristics**:

- Detailed reasoning trace with step-by-step thinking
- Clear final answer extraction
- Higher token usage for reasoning steps
- Longer response times

**Answer Extraction Logic**:

- Identifies patterns like "Therefore, the answer is...", "Final answer:", etc.
- Separates reasoning steps from final answer
- Handles cases where separation is unclear

## Future Agent Specifications

### 3. Program of Thought Agent (Planned)

**Concept**: Code-generation approach for mathematical and logical problems

- Generate Python code to solve problems systematically
- Execute code in sandboxed environment
- Return both code trace and computational result

### 4. Reasoning as Planning Agent (Planned)

**Concept**: Strategic planning with goal decomposition

- Break complex problems into subgoals
- Create execution plans for each subgoal
- Combine subgoal results systematically

### 5. Reflection Agent (Planned)

**Concept**: Self-evaluation and iterative improvement

- Generate initial answer with reasoning
- Reflect on answer quality and validity
- Generate refined answer based on reflection

## Agent Factory Pattern

```python
class ReasoningAgentServiceFactory:
    def create_service(self, agent_type: str) -> ReasoningAgentService
    def get_available_types(self) -> List[str]
    def validate_agent_type(self, agent_type: str) -> bool
    def register_agent(self, agent_type: str, agent_class: Type[ReasoningAgentService])
```

**Core Registrations**:

- "none" → NoneAgent
- "chain_of_thought" → ChainOfThoughtAgent

**Extensibility**: New agents can be registered dynamically

## Configuration Specifications

### AgentConfig Parameters

**Model Parameters** (LLM-specific):

- `temperature`: Response randomness (0.0-2.0)
- `max_tokens`: Maximum response length
- `top_p`: Nucleus sampling parameter

**Reasoning Parameters** (Agent-specific):

- **None Agent**: No specific parameters required
- **Chain of Thought**: No specific parameters required

**Example Configurations**:

```python
none_config = AgentConfig(
    agent_type="none",
    model_name="anthropic/claude-3-sonnet",
    model_parameters={"temperature": 0.7, "max_tokens": 200},
    agent_parameters={}
)

cot_config = AgentConfig(
    agent_type="chain_of_thought",
    model_name="anthropic/claude-3-sonnet",
    model_parameters={"temperature": 0.8, "max_tokens": 1000},
    agent_parameters={}
)
```

## Error Handling Categories

**Agent-Specific Errors**:

- `REASONING_EXTRACTION_FAILED`: Cannot separate reasoning from answer
- `ANSWER_SEPARATION_FAILED`: No clear final answer found
- `CODE_EXECUTION_FAILED`: For Program of Thought agent failures
- `REFLECTION_LOOP_EXCEEDED`: For Reflection agent timeouts

**Recovery Strategies**:

- Reasoning extraction failures: Non-recoverable, requires prompt engineering
- Answer separation failures: Potentially recoverable with different prompting
- API failures: Mapped to standard FailureReason categories

## Performance Characteristics

| Agent Type         | Token Usage     | Response Time | Complexity |
| ------------------ | --------------- | ------------- | ---------- |
| None               | 50-150 tokens   | 1-3 seconds   | Low        |
| Chain of Thought   | 200-800 tokens  | 3-8 seconds   | Medium     |
| Program of Thought | 300-1200 tokens | 5-15 seconds  | High       |

## Extensibility Framework

**Custom Agent Requirements**:

1. Implement `ReasoningAgentService` interface
2. Define unique `agent_type` identifier
3. Specify required reasoning parameters
4. Implement configuration validation
5. Handle agent-specific error conditions

**Registration Pattern**:

```python
factory = ReasoningAgentServiceFactory()
factory.register_agent("custom_approach", CustomAgent)
```

## Testing Considerations

**Validation Patterns**:

- **Structure over Content**: Validate response format rather than exact text
- **Pattern Recognition**: Check reasoning traces contain expected elements
- **Answer Extraction**: Verify clean answer separation from reasoning
- **Error Classification**: Ensure failure reasons are properly categorized

**Key Test Scenarios**:

- Agent consistency across multiple runs
- Reasoning pattern validation for each agent type
- Configuration validation and error handling
- Response format handling for various LLM outputs

## See Also

- **[Domain Model](v2-domain-model.md)** - ReasoningAgentService interface and domain concepts
- **[Application Services Architecture](v2-application-services-architecture.md)** - Integration with evaluation orchestration
- **[Infrastructure Requirements](v2-infrastructure-requirements.md)** - LLM client dependencies
- **[Core Behaviors](v2-core-behaviour-definition.md)** - User workflows utilizing reasoning agents
- **[Testing Strategy](v2-testing-strategy.md)** - Agent testing patterns and validation approaches

---

_This specification provides architectural guidance for reasoning agent implementation, focusing on behavioral contracts and extensibility patterns rather than detailed implementation code._
