# ML Agents v2 Data Model Specification

**Version:** 1.0
**Date:** 2025-09-17
**Purpose:** Define data structures before SQL schema implementation

## Core Data Entities

### Evaluation (Aggregate Root)

```python
Evaluation {
    evaluation_id: UUID                    # Primary key
    agent_config: AgentConfig              # Embedded value object
    preprocessed_benchmark_id: UUID        # Foreign key reference
    status: EvaluationStatus               # Enum: pending, running, completed, failed, interrupted
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    results: EvaluationResults | None      # Complex object when completed
    failure_reason: FailureReason | None   # Complex object when failed
}
```

**Storage Decisions:**

- `agent_config`: Store as JSON to preserve value object semantics
- `results`: Store as JSON blob for flexibility, extract key metrics for indexing
- `failure_reason`: Store as JSON with category field extracted for querying

### PreprocessedBenchmark (Aggregate Root)

```python
PreprocessedBenchmark {
    benchmark_id: UUID                     # Primary key
    name: str                              # Unique constraint
    description: str
    questions: List[Question]              # Stored as JSON array
    metadata: Dict[str, Any]               # JSON object
    created_at: datetime
    question_count: int                    # Derived field for performance
    format_version: str                    # For evolution support
}
```

**Storage Decisions:**

- `questions`: JSON array, indexed by question_id for retrieval
- `metadata`: JSON object for extensibility
- `question_count`: Denormalized for quick access

## Value Objects

### AgentConfig

```python
AgentConfig {
    agent_type: str                        # none, cot, pot, etc.
    model_provider: str                    # openrouter
    model_name: str                        # claude-3-sonnet, gpt-4, etc.
    model_parameters: Dict[str, Any]       # temperature, max_tokens, etc.
    agent_parameters: Dict[str, Any]   # agent-specific config
}
```

**Storage Strategy:** Embedded JSON in evaluations table with hash for deduplication queries

### Question

```python
Question {
    id: str                                # Question identifier within benchmark
    text: str                              # Question content
    expected_answer: str                   # Ground truth
    metadata: Dict[str, Any]               # Optional: difficulty, category, etc.
}
```

### Answer

```python
Answer {
    extracted_answer: str                  # Final answer
    reasoning_trace: ReasoningTrace        # Structured reasoning info
    confidence: float | None               # Optional confidence score
    execution_time: float                  # Seconds
    token_usage: TokenUsage                # Token consumption metrics
    raw_response: str                      # Full model response
}
```

### EvaluationResults

```python
EvaluationResults {
    total_questions: int
    correct_answers: int
    accuracy: float                        # Computed field
    average_execution_time: float
    total_tokens: int
    error_count: int
    detailed_results: List[QuestionResult] # Per-question outcomes
    summary_statistics: Dict[str, Any]     # Additional metrics
}
```

### QuestionResult

```python
QuestionResult {
    question_id: str
    question_text: str
    expected_answer: str
    actual_answer: str
    is_correct: bool
    reasoning_trace: ReasoningTrace
    execution_time: float
    token_usage: TokenUsage
    error: str | None                      # If processing failed
}
```

### ReasoningTrace

```python
ReasoningTrace {
    approach_type: str                     # none, cot, pot, etc.
    reasoning_text: str                    # Empty for "none", steps for others
    logprob_confidence: float | None       # When supported by model
}
```

### FailureReason

```python
FailureReason {
    category: FailureCategory              # Enum for querying
    description: str                       # Human-readable
    technical_details: str                 # Raw error info
    occurred_at: datetime
    recoverable: bool                      # Whether retry might work
}

FailureCategory = Enum[
    "parsing_error",
    "token_limit_exceeded",
    "content_guardrail",
    "model_refusal",
    "network_timeout",
    "unknown"
]
```

### TokenUsage

```python
TokenUsage {
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
}
```

## Data Relationships and Constraints

### Primary Relationships

- `Evaluation` → `PreprocessedBenchmark` (many-to-one)
- `Evaluation` contains `AgentConfig` (embedded)
- `Evaluation` contains `EvaluationResults` (embedded when completed)
- `Evaluation` contains `FailureReason` (embedded when failed)

### Key Constraints

1. **Evaluation Status Logic:**

   - `results` must be NULL unless `status = 'completed'`
   - `failure_reason` must be NULL unless `status = 'failed'`
   - `started_at` must be NULL if `status = 'pending'`
   - `completed_at` must be NULL unless `status IN ('completed', 'failed')`

2. **AgentConfig Validation:**

   - `model_parameters` must be valid JSON object
   - `agent_parameters` must be valid JSON object
   - `agent_type` must be in supported types list

3. **Benchmark Integrity:**
   - `name` must be unique across all benchmarks
   - `question_count` must equal `len(questions)`
   - All questions must have non-empty `text` and `expected_answer`

## Indexing Strategy

### Performance-Critical Queries

```sql
-- Find evaluations by status
CREATE INDEX idx_evaluations_status ON evaluations(status);

-- Find evaluations by benchmark
CREATE INDEX idx_evaluations_benchmark ON evaluations(preprocessed_benchmark_id);

-- Find benchmarks by name
CREATE UNIQUE INDEX idx_benchmarks_name ON preprocessed_benchmarks(name);

-- Query evaluations by agent configuration similarity
CREATE INDEX idx_evaluations_agent_hash ON evaluations(agent_config_hash);
```

### JSON Field Indexing (SQLite 3.45+ / PostgreSQL)

```sql
-- Index failure categories for filtering
CREATE INDEX idx_evaluations_failure_category
ON evaluations(json_extract(failure_reason, '$.category'));

-- Index agent types for analysis
CREATE INDEX idx_evaluations_agent_type
ON evaluations(json_extract(agent_config, '$.agent_type'));
```

## Data Size Estimates

### Typical Object Sizes

- `AgentConfig`: ~200-500 bytes (JSON)
- `Question`: ~100-2000 bytes (varies by benchmark)
- `Answer`: ~500-5000 bytes (depends on reasoning length)
- `EvaluationResults`: ~10KB-1MB (depends on question count and detail level)

### Storage Projections

- 1000 evaluations × 100KB average = ~100MB
- 50 benchmarks × 10MB average = ~500MB
- Total: ~600MB for substantial research dataset

## Migration and Evolution Strategy

### Schema Versioning

- `format_version` field in benchmarks supports data migration
- JSON fields provide flexibility for adding new attributes
- Separate migration scripts for structural changes

### Backward Compatibility

- New reasoning approaches add to existing enums
- Additional metrics extend JSON objects without breaking existing queries
- Agent config evolution through versioned parameter schemas

## Data Integrity Patterns

### Transactional Boundaries

- Evaluation creation, execution, and completion as separate transactions
- Benchmark preprocessing as atomic operation
- Results storage with optimistic concurrency control

### Validation Layers

1. **Domain Level:** Entity and value object invariants
2. **Application Level:** Cross-aggregate consistency
3. **Database Level:** Constraints and triggers
4. **API Level:** Input validation and sanitization

## See Also

- **[Domain Model](v2-domain-model.md)** - Business entity definitions implemented in this data model
- **[Infrastructure Requirements](v2-infrastructure-requirements.md)** - Database technology and configuration details
- **[Application Services Architecture](v2-application-services-architecture.md)** - Service patterns using this data model
- **[Project Structure](v2-project-structure.md)** - Repository implementation organization
