# Ubiquitous Language

**Version:** 1.0
**Date:** 2025-09-17
**Purpose:** Shared vocabulary between domain experts and developers

## Core Domain Concepts

### Evaluation

A research experiment that applies a specific reasoning approach to all questions in a benchmark. Has lifecycle states: pending → running → completed/failed/interrupted.

### Agent Configuration (AgentConfig)

Reusable specification of how reasoning should be performed: agent type, model, and parameters. Multiple evaluations can share the same configuration.

### Preprocessed Benchmark

Standardized dataset of questions ready for evaluation. Immutable once created. Contains questions with expected answers.

### Reasoning Agent

Service that applies a specific reasoning approach (none, chain-of-thought) to individual questions using a language model.

### Question

Individual test case from a benchmark with text, expected answer, and optional metadata (difficulty, category).

### Answer

Complete response from a reasoning agent including extracted answer, reasoning trace, timing, and token usage.

## Reasoning Approaches

### None (Direct)

Reasoning approach where the model answers directly without explicit reasoning steps.

### Chain of Thought (CoT)

Reasoning approach where the model shows step-by-step thinking before providing the final answer.

## Evaluation Lifecycle

### Pending

Evaluation is created and configured but not yet started.

### Running

Evaluation is actively processing questions, showing real-time progress.

### Completed

All questions processed successfully, results available for analysis.

### Failed

Evaluation could not start due to system errors (authentication, configuration).

### Interrupted

User stopped evaluation mid-execution (Ctrl+C), partial results preserved.

## Results and Analysis

### Evaluation Results

Complete outcome metrics: accuracy, timing, token usage, and per-question details.

### Reasoning Trace

Documentation of the reasoning process used for each question (empty for "none", step-by-step text for "cot").

### Failure Reason

Categorized explanation of why individual questions or entire evaluations failed (parsing errors, timeouts, etc.).

## Technical Terms

### Model Provider

External service providing language model access (OpenRouter, LiteLLM).

### Model Parameters

Configuration affecting model behavior: temperature, max tokens, etc.

### Token Usage

Count of input/output tokens consumed by language model API calls.

### Progress Tracking

Real-time display of evaluation execution: questions completed, current question, estimated time.

## Research Context

### Benchmark

Dataset of questions used to evaluate reasoning approaches. Examples: GPQA (scientific reasoning), FOLIO (logic).

### Research Question

High-level inquiry driving evaluation design: "Do all tasks benefit from reasoning?", "How do different approaches compare?"

### Task-Approach Fit

Research concept examining whether certain reasoning approaches work better for specific types of questions.

## System Operations

### Synchronous Execution

Evaluation runs with CLI blocking and showing real-time progress until completion.

### Background Processing

Internal async processing of individual questions while maintaining synchronous user interface.

### Health Check

System verification ensuring database connectivity, API access, and benchmark availability.

## Error Categories

### Configuration Error

Invalid agent types, models, or parameters preventing evaluation creation.

### Resource Error

Missing benchmarks, evaluations, or API connectivity issues.

### Runtime Error

Problems during evaluation execution: authentication, timeouts, interruptions.

## User Interface Terms

### Command

CLI operation like `evaluate create`, `benchmark list`, `health`.

### Evaluation ID

Short identifier for created evaluations (eval_456) used in subsequent commands.

### Short Name

User-friendly benchmark identifier (GPQA) mapping to full dataset name (BENCHMARK-01-GPQA.csv).

### Real-time Progress

Live display showing question processing status during evaluation execution.

---

## Usage Guidelines

**Consistent Terminology**: All team members use identical terms for the same concepts.

**Context Clarity**: Distinguish between domain concepts (Evaluation) and technical implementations (EvaluationModel).

**Precision**: Use specific terms rather than generic ones (AgentConfig vs "settings", Reasoning Trace vs "output").

**Shared Understanding**: Researchers and developers use the same vocabulary when discussing system behavior.

## See Also

- **[Domain Model](v2-domain-model.md)** - Detailed definitions of entities and concepts referenced in this vocabulary
- **[Core Behaviors](v2-core-behaviour-definition.md)** - User workflows using this shared language
- **[CLI Design](v2-cli-design.md)** - Command interface reflecting this terminology
