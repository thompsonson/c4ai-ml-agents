# ML Agents v2 Domain Model

**Version:** 1.1
**Date:** 2025-09-17
**Status:** Draft

## Overview

This document defines the domain model for the ML Agents v2 reasoning research platform. The model follows Domain-Driven Design principles to create clear boundaries between business concepts and technical implementation.

## Domain Context

**Business Domain**: Reasoning Research Platform
**Core Purpose**: Evaluate the effectiveness of different AI reasoning approaches across various benchmarks

**Key Research Questions**:

1. Do all tasks benefit from reasoning?
2. Do different models show varying benefits from reasoning?
3. How do different reasoning approaches compare?
4. Is there a task-approach fit?

## Domain Entities

### Aggregate Roots

#### Evaluation

**Purpose**: Orchestrates the evaluation of an AgentConfig against a PreprocessedBenchmark

**Identity**: EvaluationId (unique identifier)

**Attributes**:

- `evaluation_id`: Unique identifier
- `agent_config`: Configuration for reasoning approach (Value Object)
- `preprocessed_benchmark_id`: Reference to benchmark data
- `status`: Evaluation lifecycle state (pending, running, completed, failed, interrupted)
- `created_at`: When evaluation was initiated
- `started_at`: When execution began
- `completed_at`: When execution finished
- `results`: Evaluation outcomes and metrics
- `failure_reason`: Detailed failure information when status is 'failed'

**Business Rules**:

- Cannot be modified once status is 'completed' or 'failed'
- Must have valid AgentConfig and PreprocessedBenchmark reference
- Results can only be set when status transitions to 'completed'
- Failure reason must be provided when status transitions to 'failed'

**Lifecycle States**:

```
pending → running → completed
              ↓         ↓
            failed  interrupted
```

**Methods**:

- `start_execution()`: Transition to running state
- `complete_with_results(results: EvaluationResults)`: Complete evaluation
- `fail_with_reason(failure_reason: FailureReason)`: Mark as failed with specific reason
- `can_be_modified()`: Check if evaluation can be changed

---

#### PreprocessedBenchmark

**Purpose**: Ready-to-evaluate dataset with standardized format

**Identity**: BenchmarkId (unique identifier)

**Attributes**:

- `benchmark_id`: Unique identifier
- `name`: Human-readable benchmark name (e.g., "BHEH Logical Reasoning")
- `description`: Benchmark description and purpose
- `questions`: Collection of standardized questions
- `metadata`: Additional benchmark information
- `created_at`: When preprocessing completed
- `question_count`: Total number of questions
- `format_version`: Data format version for compatibility

**Business Rules**:

- Immutable once preprocessing is complete
- All questions must follow standardized format
- Must have at least one question
- Name must be unique within system

**Methods**:

- `get_questions()`: Return all questions for evaluation
- `get_sample(size: int)`: Return random sample of questions
- `get_metadata()`: Return benchmark metadata

### Domain Services

#### ReasoningAgentService

**Purpose**: Stateless service that applies reasoning logic to questions

**Interface**:

```python
class ReasoningAgentService:
    def process_question(self, question: Question, config: AgentConfig) -> Answer
    def get_agent_type(self) -> str
    def validate_config(self, config: AgentConfig) -> bool
```

**Initial Implementations** (focus on 2 core approaches):

1. **None**: Direct prompting without reasoning steps
2. **ChainOfThought**: Step-by-step reasoning process

**Future Implementations** (pending paper references):

3. **ProgramOfThought**: Code-based problem solving
4. **ReasoningAsPlanning**: Strategic planning with goal decomposition
5. **Reflection**: Self-evaluation and iterative improvement
6. **ChainOfVerification**: Systematic verification with follow-up questions
7. **SkeletonOfThought**: Hierarchical outline-first reasoning
8. **TreeOfThought**: Multiple reasoning path exploration and synthesis

### Value Objects

#### AgentConfig

**Purpose**: Configuration that determines which ReasoningAgent to use and how

This is a value object because configurations are reusable and shareable across evaluations. Two AgentConfig instances with identical attributes represent the same conceptual configuration, regardless of when or where they were created. The non-deterministic nature of LLM outputs means the same configuration can produce different results, but that variation belongs in the EvaluationResults, not the configuration itself.

**Attributes**:

- `agent_type`: Type of reasoning agent (None, CoT, PoT, etc.)
- `model_provider`: LLM provider (Anthropic, OpenRouter, Cohere, etc.)
- `model_name`: Specific model identifier
- `model_parameters`: Model-specific parameters (temperature, max_tokens, etc.)
- `agent_parameters`: Agent-specific parameters

**Business Rules**:

- Must specify valid agent_type from available implementations
- Model parameters must be compatible with selected provider
- Reasoning parameters must be valid for selected agent type

**Methods**:

- `get_reasoning_agent_service()`: Factory method for service
- `validate_configuration()`: Ensure configuration is valid
- `to_dict()`: Serialize for storage/comparison
- `equals(other: AgentConfig)`: Value-based equality comparison

**Immutable**: Yes

---

#### Question

**Purpose**: Standardized input data from benchmark

**Attributes**:

- `id`: Question identifier within benchmark
- `text`: Question content
- `expected_answer`: Ground truth answer
- `metadata`: Additional question information (difficulty, category, etc.)

**Immutable**: Yes

---

#### Answer

**Purpose**: Response from reasoning agent with trace information

**Attributes**:

- `extracted_answer`: Clean, final answer
- `reasoning_trace`: Step-by-step reasoning process
- `confidence`: Agent confidence score (if available)
- `execution_time`: Time taken to generate answer
- `token_usage`: LLM token consumption metrics
- `raw_response`: Full model response text

**Immutable**: Yes

---

#### EvaluationResults

**Purpose**: Complete evaluation outcomes and metrics

**Attributes**:

- `total_questions`: Number of questions evaluated
- `correct_answers`: Number of correctly answered questions
- `accuracy`: Percentage accuracy (correct/total)
- `average_execution_time`: Mean time per question
- `total_tokens`: Total token consumption
- `error_count`: Number of failed questions
- `detailed_results`: Per-question results
- `summary_statistics`: Additional metrics

**Immutable**: Yes

**Methods**:

- `calculate_accuracy()`: Compute accuracy percentage
- `get_performance_summary()`: Summary statistics
- `export_detailed_csv()`: Export for analysis

---

#### ReasoningTrace

**Purpose**: Reasoning documentation

**Attributes**:

- `approach_type`: Type of reasoning used ("None" or "ChainOfThought")
- `reasoning_text`: The reasoning content (empty for "None", step-by-step text for "ChainOfThought")
- `logprob_confidence`: Optional confidence score from model logprobs (when supported)

**Immutable**: Yes

**Notes**:

- For "None" approach: `reasoning_text` is empty string
- For "ChainOfThought" approach: `reasoning_text` contains the model's reasoning steps
- `logprob_confidence` uses libraries like structured-logprobs when model supports token-level confidence

---

#### FailureReason

**Purpose**: Detailed categorization of evaluation failures

Understanding why evaluations fail is crucial for researchers to improve their approaches and identify systematic issues.

**Attributes**:

- `category`: Type of failure (parsing_error, token_limit_exceeded, content_guardrail, model_refusal, network_timeout, rate_limit_exceeded, credit_limit_exceeded, authentication_error, unknown)
- `description`: Human-readable failure description
- `technical_details`: Raw error information for debugging
- `occurred_at`: When the failure occurred
- `recoverable`: Whether this failure type might succeed on retry

**Subtypes**:

- `StructuredOutputParsingFailure`: Model response doesn't match expected format
- `TokenLimitExceeded`: Request exceeded model's token capacity
- `ContentGuardrailTriggered`: Model safety systems prevented response
- `ModelRefusal`: Model explicitly declined to answer
- `NetworkTimeout`: Communication failure with model provider
- `RateLimitExceeded`: API rate limit reached, retry after delay
- `CreditLimitExceeded`: Insufficient API credits or budget
- `AuthenticationError`: Invalid API key or authentication failure
- `UnknownFailure`: Unexpected error not fitting other categories

**Immutable**: Yes

**Methods**:

- `is_recoverable()`: Whether retry might succeed
- `get_category_description()`: Human-friendly explanation of failure type

## Domain Relationships

```
Evaluation (Aggregate Root)
├── AgentConfig (Value Object)
│   └── → ReasoningAgentService (Domain Service)
│       └── → Question → Answer (Value Objects)
├── → PreprocessedBenchmark (Reference)
│   └── Questions (Value Objects)
├── EvaluationResults (Value Object)
│   └── ReasoningTrace (Value Object)
└── FailureReason (Value Object, when failed)
```

### Relationship Rules

1. **Evaluation contains AgentConfig**: 1:1 relationship, but AgentConfig can be reused across multiple evaluations
2. **Evaluation references PreprocessedBenchmark**: 1:1 relationship, benchmark exists independently
3. **AgentConfig specifies ReasoningAgentService**: Configuration determines which service implementation to use
4. **ReasoningAgentService processes Questions**: Service is stateless, processes questions individually
5. **EvaluationResults aggregates Answers**: Results contain outcomes from all processed questions
6. **FailureReason provides failure context**: Only present when evaluation fails, explains what went wrong

## Bounded Context Boundaries

### Core Domain

- **Entities**: Evaluation, PreprocessedBenchmark
- **Services**: ReasoningAgentService
- **Value Objects**: AgentConfig, Question, Answer, EvaluationResults, ReasoningTrace, FailureReason

### Supporting Domains

- **Infrastructure**: Database persistence, API clients, file system
- **Application**: CLI commands, orchestration services
- **Reasoning Implementations**: Concrete reasoning agent services

## Domain Events (Future Consideration)

Potential domain events for future implementation:

- `EvaluationStarted`
- `EvaluationCompleted`
- `EvaluationFailed`
- `QuestionProcessed`
- `BenchmarkPreprocessed`

## Invariants and Business Rules

### System-Wide Invariants

1. All evaluations must have unique identifiers
2. PreprocessedBenchmarks are immutable after creation
3. Evaluation results can only be set once
4. AgentConfigs must be valid for their specified reasoning approach
5. Failed evaluations must have a FailureReason

### Evaluation Invariants

1. Cannot modify evaluation once completed or failed
2. Results must match the number of questions in benchmark
3. Execution time must be positive
4. Status transitions must follow defined lifecycle
5. FailureReason is required when status is 'failed'

### PreprocessedBenchmark Invariants

1. Must contain at least one question
2. All questions must have expected answers
3. Benchmark names must be unique
4. Cannot be modified after creation

### AgentConfig Invariants (Value Object)

1. All required fields must be present
2. Model parameters must be valid for the specified provider
3. Reasoning parameters must be compatible with agent type
4. Equality is determined by attribute values, not identity

## Repository Interfaces

```python
class EvaluationRepository:
    def save(self, evaluation: Evaluation) -> None
    def get_by_id(self, evaluation_id: EvaluationId) -> Evaluation
    def get_by_status(self, status: EvaluationStatus) -> List[Evaluation]
    def get_by_agent_config(self, config: AgentConfig) -> List[Evaluation]
    def get_all(self) -> List[Evaluation]

class PreprocessedBenchmarkRepository:
    def save(self, benchmark: PreprocessedBenchmark) -> None
    def get_by_id(self, benchmark_id: BenchmarkId) -> PreprocessedBenchmark
    def get_by_name(self, name: str) -> PreprocessedBenchmark
    def get_all() -> List[PreprocessedBenchmark]
```

## Factory Patterns

### ReasoningAgentServiceFactory

```python
class ReasoningAgentServiceFactory:
    def create_service(self, agent_type: str) -> ReasoningAgentService
    def get_available_types(self) -> List[str]
    def validate_agent_type(self, agent_type: str) -> bool
```

## Domain Model Evolution

### Version 1.1 (Current)

- Core aggregates: Evaluation, PreprocessedBenchmark
- AgentConfig as reusable value object
- Rich failure taxonomy with FailureReason
- 8+ reasoning approaches planned
- SQLite persistence
- CLI interface

### Future Considerations

- Domain events for real-time monitoring
- PreprocessedBenchmark versioning to support benchmark evolution
- Experiment batching and scheduling
- Advanced metrics and reporting
- Multi-user access and permissions

---

## Implementation Notes

### Technology Alignment

- **Language**: Python 3.9+
- **Persistence**: SQLite for development, extensible to PostgreSQL
- **API Integration**: Multiple LLM providers (Anthropic, OpenRouter, Cohere)
- **Testing**: pytest with domain-focused test structure

### File Organization

Domain model classes should be organized by aggregate:

- `core/domain/evaluation/` - Evaluation aggregate
- `core/domain/benchmark/` - PreprocessedBenchmark aggregate
- `core/domain/value_objects/` - Shared value objects
- `core/domain/services/` - Domain service interfaces

### Validation Strategy

- Domain entities validate their own state
- Repository implementations handle persistence validation
- Application services coordinate cross-aggregate validation
- Value objects are immutable and validate on construction
- AgentConfig equality based on value comparison, not identity

### Key Design Rationale

**AgentConfig as Value Object**: This design decision reflects that configurations are conceptual templates that can be shared and reused. Multiple researchers can use the same reasoning approach configuration, and the system recognizes these as the same thing rather than separate entities. The non-deterministic nature of LLM outputs means identical configurations can produce different results, but this variation is captured in EvaluationResults, not in the configuration itself.

**Rich Failure Taxonomy**: Research platforms must help users understand why experiments fail. Generic "failed" status provides little actionable information, while specific failure categories help researchers identify whether issues stem from configuration problems, infrastructure limitations, or model behavior.

**Immutable Benchmarks**: Ensuring benchmark consistency is crucial for research validity. While future versions might support benchmark versioning, the current model prioritizes research integrity by preventing accidental benchmark modifications that could invalidate comparative analysis.

---

## See Also

- **[Data Model](v2-data-model.md)** - Database schema implementation of these domain entities
- **[Application Services Architecture](v2-application-services-architecture.md)** - Service coordination patterns using these domain concepts
- **[Ubiquitous Language](v2-ubiquitous-language.md)** - Shared vocabulary definitions for domain terms
- **[Core Behaviors](v2-core-behaviour-definition.md)** - User workflows that manipulate these domain entities

---

_This domain model serves as the foundation for all v2 implementation decisions and should be referenced when making architectural choices._
