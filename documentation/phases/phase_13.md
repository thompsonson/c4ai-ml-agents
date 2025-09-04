# Phase 13: Local API Testing with None Approach (Active)

## Strategic Context

**Purpose**: Validate the None baseline reasoning approach against the local pop-os API server using the Qwen/Qwen2.5-1.5B-Instruct model to establish baseline performance metrics.

**Testing Target**: `http://pop-os:8000/v1/chat/completions` with OpenAI-compatible API format

**Model**: `Qwen/Qwen2.5-1.5B-Instruct` (verified working via curl test)

**Status**: 🚧 **IN PROGRESS** - Setting up test runs

## Implementation Plan

### **Phase 13.1: API Configuration Setup**

**Objective**: Configure ml-agents to use the local pop-os API endpoint

**Configuration Requirements**:

- Base URL: `http://pop-os:8000/v1`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Provider: Custom (OpenAI-compatible)
- Authentication: None required (local server)

**Implementation Steps**:

1. Verify local API configuration options in ml-agents
2. Set up environment variables for local testing
3. Test connection with health check

### **Phase 13.2: None Approach Test Runs**

**Objective**: Execute baseline None reasoning approach against test benchmarks

**Test Matrix**:

- **Approach**: None (baseline)
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Endpoint**: <http://pop-os:8000/v1>
- **Benchmarks**: Small test benchmarks for validation
- **Sample Sizes**: 5, 10, 25 samples for performance testing

**Expected Outputs**:

- Response times and token usage metrics
- Baseline accuracy measurements
- API stability and error rates
- Local processing performance benchmarks

### **Phase 13.3: Performance Validation**

**Objective**: Validate local API performance characteristics

**Metrics to Capture**:

- **Response Latency**: Time per API call
- **Token Processing**: Input/output token rates
- **Throughput**: Requests per minute sustainable rate
- **Error Rates**: Failed requests and timeouts
- **Resource Usage**: Local system impact

**Success Criteria**:

- Stable API responses without errors
- Consistent response times under 10 seconds
- Token usage tracking functional
- Baseline accuracy established

## Test Commands

### **Environment Setup**

```bash
# Set local API configuration
export ML_AGENTS_API_BASE_URL="http://pop-os:8000/v1"
export ML_AGENTS_API_KEY=""  # Not required for local
export ML_AGENTS_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export ML_AGENTS_PROVIDER="local-openai"
```

### **API Health Check**

```bash
# Test basic API connectivity (verified working)
curl http://pop-os:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Who won the world series in 2020?"}
    ]
  }' | jq
```

### **ML Agents Test Runs**

```bash
# Test with minimal benchmark samples
ml-agents eval run TEST_BENCHMARK None --samples 5 \
  --provider local-openai \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --api-base "http://pop-os:8000/v1"

# Performance test with larger sample
ml-agents eval run TEST_BENCHMARK None --samples 25 \
  --provider local-openai \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --api-base "http://pop-os:8000/v1"
```

### **Results Analysis**

```bash
# Check latest experiment results
ml-agents results list --status completed

# Export results for analysis
ml-agents results export <EXPERIMENT_ID> --format json

# View performance summary
ml-agents results analyze <EXPERIMENT_ID> --type performance
```

## Expected Results

### **Baseline Performance Profile**

**Response Format (Verified)**:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1756978603,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Response text...",
      "refusal": null
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 31,
    "total_tokens": 75,
    "completion_tokens": 44
  }
}
```

**Performance Expectations**:

- **Response Time**: 2-5 seconds per request (local processing)
- **Token Rate**: ~15 tokens/second (based on 1.5B model size)
- **Accuracy**: Baseline performance for comparison with reasoning approaches
- **Stability**: 99%+ success rate on local network

### **Validation Checkpoints**

**Phase 13.1 Completion Criteria**:

- [ ] Local API connection established
- [ ] ML Agents configured for local endpoint
- [ ] Configuration validated with test requests

**Phase 13.2 Completion Criteria**:

- [ ] None approach executes successfully
- [ ] Test benchmarks complete without errors
- [ ] Performance metrics captured
- [ ] Results stored in database

**Phase 13.3 Completion Criteria**:

- [ ] Performance profile established
- [ ] Baseline metrics documented
- [ ] System stability confirmed
- [ ] Ready for reasoning approach comparisons

## Integration Benefits

### **Development Workflow**

- **Fast Iteration**: Local model eliminates API costs and rate limits
- **Offline Testing**: Complete development environment without internet dependency
- **Performance Tuning**: Direct control over model parameters and hardware
- **Privacy**: Sensitive test data remains local

### **Research Value**

- **Baseline Establishment**: None approach provides control group
- **Cost-Effective Testing**: Local processing for extensive benchmark runs
- **Reproducible Results**: Consistent local environment
- **Hardware Optimization**: Understanding of local processing capabilities

## Next Phase Integration

**Phase 14 Preparation**: Once None approach baseline is established, Phase 14 will compare reasoning approaches (ChainOfThought, TreeOfThought, etc.) against the same local API to measure reasoning effectiveness.

**Comparative Analysis**: Local None baseline will serve as the control group for measuring:

- Reasoning approach performance gains
- Token usage efficiency of different approaches
- Response quality improvements
- Processing time trade-offs

## Risk Mitigation

**API Stability**: Local server may have different behavior than production APIs

- **Mitigation**: Document any API compatibility issues for future reference

**Model Limitations**: 1.5B parameter model may have different baseline performance

- **Mitigation**: Establish relative performance comparisons rather than absolute scores

**Network Dependencies**: Local network connectivity to pop-os

- **Mitigation**: Verify network connectivity as part of setup validation

## Success Metrics

**Functional Success**:

- None approach executes without errors against local API
- All test benchmarks complete successfully
- Results properly stored in ml_agents_results.db
- Performance metrics captured accurately

**Performance Success**:

- Consistent response times < 10 seconds
- Token usage tracking functional
- No API timeout or connection errors
- Stable performance across multiple test runs

**Integration Success**:

- Results compatible with existing analysis tools
- Database integration working properly
- Export functionality operational
- Ready for comparative analysis with other approaches

This phase establishes the foundation for cost-effective local testing while providing baseline metrics for the research platform's reasoning effectiveness analysis.
