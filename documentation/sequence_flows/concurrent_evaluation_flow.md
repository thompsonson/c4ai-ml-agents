# Concurrent Evaluation Sequence Flow

This document provides a comprehensive sequence flow diagram for the ML Agents concurrent evaluation system, showing the interaction between all major classes during concurrent benchmark processing.

## Overview

The concurrent evaluation system processes multiple prompts simultaneously using async/await patterns and semaphore-based concurrency limiting. The system transforms benchmark datasets, applies reasoning approaches, makes concurrent API calls to LLM providers, extracts structured responses, and calculates accurate performance metrics.

## Complete Sequence Flow

```mermaid
sequenceDiagram
    participant CLI as eval.py<br/>run_single_experiment
    participant ConcExp as eval.py<br/>_run_concurrent_experiment
    participant Runner as ExperimentRunner
    participant Loader as BBEHDatasetLoader
    participant ReasoningEngine as BaseReasoning<br/>(ChainOfThought/None/etc)
    participant ClientMgr as InstructorClientManager
    participant APIClient as OpenAI/Anthropic<br/>Client
    participant PydanticModel as ReasoningExtraction<br/>Pydantic Models
    participant ResultProcessor as ConcurrentExperimentResult

    Note over CLI: User runs: ml-agents eval run BENCHMARK ChainOfThought --concurrent

    CLI->>CLI: Parse CLI arguments and validate config
    CLI->>Runner: Create ExperimentRunner(experiment_config)
    CLI->>ConcExp: asyncio.run(_run_concurrent_experiment())

    Note over ConcExp: Async concurrent experiment orchestration

    ConcExp->>Loader: Create BBEHDatasetLoader(runner.config)
    ConcExp->>Loader: load_dataset(benchmark_id)
    Loader->>Loader: Load dataset from HuggingFace/CSV
    Loader-->>ConcExp: Return dataset with INPUT/OUTPUT format

    ConcExp->>Loader: sample_data(dataset, sample_size)
    Loader-->>ConcExp: Return sampled dataset

    ConcExp->>ConcExp: Extract prompts and expected_answers<br/>from INPUT/OUTPUT columns

    ConcExp->>Runner: await run_reasoning_concurrent(<br/>prompts, approach, expected_answers)

    Note over Runner: Main concurrent processing orchestration

    Runner->>Runner: Validate reasoning_approach
    Runner->>Runner: Get reasoning_engine._get_or_create_approach()
    Runner-->>ReasoningEngine: Initialize specific reasoning class<br/>(ChainOfThought/None/etc)

    Runner->>ReasoningEngine: await execute_concurrent(<br/>prompts, concurrency_limit)

    Note over ReasoningEngine: Concurrent reasoning execution

    ReasoningEngine->>ReasoningEngine: Prepare enhanced prompts<br/>using _prepare_enhanced_prompt()
    ReasoningEngine->>ReasoningEngine: Get reasoning-specific extraction model<br/>from REASONING_EXTRACTION_MODELS
    ReasoningEngine->>ReasoningEngine: Convert prompts to messages_list<br/>[{"role": "user", "content": prompt}]

    ReasoningEngine->>ClientMgr: await extract_concurrent_responses(<br/>messages_list, extraction_model, concurrency_limit)

    Note over ClientMgr: Semaphore-controlled concurrent API calls

    ClientMgr->>ClientMgr: Create asyncio.Semaphore(concurrency_limit)

    loop For each message in messages_list
        ClientMgr->>ClientMgr: bounded_extract(messages, request_id)
        Note over ClientMgr: Semaphore-controlled execution
        ClientMgr->>ClientMgr: async with semaphore
        ClientMgr->>ClientMgr: extract_structured_response_async()
        ClientMgr->>APIClient: await client.chat.completions.create(<br/>model, response_model, messages)
        APIClient->>APIClient: Make HTTP request to LLM provider<br/>(vLLM/OpenAI/Anthropic)
        APIClient-->>ClientMgr: Return raw LLM response
        ClientMgr->>PydanticModel: Parse response using Instructor<br/>with reasoning-specific model
        PydanticModel->>PydanticModel: Validate and extract:<br/>- full_reasoning_text<br/>- answer_value<br/>- confidence
        PydanticModel-->>ClientMgr: Return structured extraction object
        ClientMgr-->>ClientMgr: Add to concurrent results
    end

    ClientMgr->>ClientMgr: await asyncio.gather(*tasks)
    ClientMgr-->>ReasoningEngine: Return list of extraction objects

    ReasoningEngine-->>Runner: Return concurrent responses

    Note over Runner: Result processing and accuracy calculation

    Runner->>Runner: Process responses into result_data format
    loop For each response
        Runner->>Runner: Extract answer from response.extracted_answer
        Runner->>Runner: _calculate_correctness(extracted, expected)
        Runner->>Runner: Create result_data with:<br/>- extracted_answer<br/>- expected_answer<br/>- is_correct<br/>- sample_id
    end

    Runner->>Runner: Calculate metrics:<br/>- successful_count<br/>- correct_count<br/>- accuracy percentage
    Runner-->>ConcExp: Return processed results list

    ConcExp->>ResultProcessor: Create ConcurrentExperimentResult(<br/>results, approach_name)
    ResultProcessor->>ResultProcessor: Calculate final metrics:<br/>- accuracy = correct_answers / total<br/>- results_summary<br/>- cost_summary
    ResultProcessor-->>ConcExp: Return result object

    ConcExp-->>CLI: Return experiment result

    Note over CLI: Display results and completion

    CLI->>CLI: Display experiment results table<br/>showing real accuracy metrics
    CLI->>CLI: Display experiment completion info<br/>with duration and cost
```

## Key Components

### 1. CLI Layer (`eval.py`)
- **Entry Point**: `run_single_experiment()` - Handles command parsing and orchestration
- **Concurrent Orchestrator**: `_run_concurrent_experiment()` - Async function managing the complete concurrent flow

### 2. Dataset Management (`BBEHDatasetLoader`)
- **Data Loading**: Loads benchmarks from HuggingFace Hub or local CSV files
- **Sampling**: Applies sample size limits as specified by `--samples` parameter
- **Format Standardization**: Ensures INPUT/OUTPUT column format for consistent processing

### 3. Experiment Orchestration (`ExperimentRunner`)
- **Process Coordination**: `run_reasoning_concurrent()` - Main async method coordinating concurrent execution
- **Reasoning Engine Management**: Creates and manages reasoning approach instances
- **Result Processing**: Handles response processing, accuracy calculation, and result compilation

### 4. Reasoning Layer (`BaseReasoning` + Subclasses)
- **Approach-Specific Logic**: Each reasoning class (ChainOfThought, None, etc.) implements specific prompt enhancement
- **Concurrent Execution**: `execute_concurrent()` - Manages prompt preparation and concurrent API calls
- **Prompt Enhancement**: `_prepare_enhanced_prompt()` - Adds reasoning-specific instructions to prompts

### 5. Client Management (`InstructorClientManager`)
- **Provider Abstraction**: Handles different LLM providers (OpenAI, Anthropic, Cohere, local-openai)
- **Concurrent Control**: Uses asyncio.Semaphore to limit concurrent API requests
- **Structured Extraction**: Integrates with Instructor library for reliable response parsing

### 6. Structured Extraction (Pydantic Models)
- **Response Parsing**: Reasoning-specific Pydantic models extract structured data from LLM responses
- **Data Validation**: Ensures consistent format with `answer_value`, `full_reasoning_text`, `confidence`
- **Compatibility**: Provides `extracted_answer` property for backward compatibility

### 7. Result Processing (`ConcurrentExperimentResult`)
- **Metrics Calculation**: Computes accurate accuracy by comparing extracted vs expected answers
- **Result Compilation**: Aggregates individual results into experiment summary
- **Performance Tracking**: Tracks timing, costs, and success rates

## Flow Highlights

### Concurrent Processing
- Uses `asyncio.Semaphore` to limit concurrent requests (default: 10)
- Each API request is bounded by semaphore to prevent overwhelming the LLM provider
- Concurrent operations are gathered using `asyncio.gather()` for parallel execution

### Accuracy Calculation
- **Fixed Issue**: No longer uses placeholder 100% accuracy
- **Real Comparison**: Compares `extracted_answer` vs `expected_answer` using case-insensitive matching
- **Realistic Metrics**: Shows actual model performance on challenging benchmarks like GPQA

### Error Handling
- Graceful handling of API failures with None returns
- Partial results support - failed requests don't stop the entire experiment
- Comprehensive logging for debugging concurrent operations

### Data Flow
1. **Dataset** → INPUT/OUTPUT format standardization
2. **Prompts** → Reasoning-specific enhancement
3. **API Calls** → Concurrent execution with semaphore limiting
4. **Responses** → Structured extraction using Pydantic models
5. **Results** → Accuracy calculation and metric compilation

This architecture enables efficient, scalable evaluation of reasoning approaches across large benchmarks while maintaining accuracy and reliability.
