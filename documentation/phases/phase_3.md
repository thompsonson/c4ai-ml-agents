## Phase 3 Implementation Decisions

### **Product Strategy**
1. **Implementation Order**: Build 3-4 approaches end-to-end first (None, CoT, PoT, Reflection) before adding others. This enables faster validation and user feedback.

2. **User Experience**: Keep reasoning approaches pluggable with minimal additional configuration. Use the existing `ExperimentConfig` parameter structure.

3. **Performance Trade-offs**: Prioritize research quality over speed. Add configurable depth limits (max_iterations, max_steps) for cost control.

### **Architecture Decisions**
4. **Prompt Strategy**: Hybrid approach - base prompt templates with reasoning-specific extensions. Create `src/reasoning/prompts/` directory.

5. **Response Parsing**: Each approach handles its own parsing but returns standardized `StandardResponse` with reasoning-specific metadata in the `metadata` field.

6. **Multi-step Reasoning**: Sequential calls with configurable iteration limits. Store intermediate results in `StandardResponse.metadata['steps']`.

7. **Error Handling**: Return partial results with error details in metadata. Continue research even with partial failures.

### **Technical Implementation**
8. **State Management**: In-memory state objects for simplicity. Design interfaces for future persistence hooks.

9. **Concurrency**: Start synchronous, design base classes for future async support.

10. **Testing**: Continue comprehensive mocking. Add optional integration tests with `@pytest.mark.integration` decorator.

11. **Evaluation Metrics**: Enhance `StandardResponse.metadata` with:
   - `reasoning_steps`: number of steps
   - `verification_count`: for CoVe
   - `approach_specific_metrics`: custom per approach

12. **Baseline**: Keep "None" minimal for true baseline measurement.

### **Implementation Pattern**
```python
class BaseReasoning:
    def execute(self, prompt: str) -> StandardResponse:
        # Standard interface all approaches implement
        pass

class ChainOfThoughtReasoning(BaseReasoning):
    def execute(self, prompt: str) -> StandardResponse:
        enhanced_prompt = self._build_cot_prompt(prompt)
        response = self.client.generate(enhanced_prompt)
        response.metadata['reasoning_steps'] = self._count_steps(response.text)
        return response
```

This provides clear guidance while maintaining flexibility for research needs.
