**ATDD Testing Strategy:**

**Acceptance Scenarios** based on your 3 core behaviors:

- Researcher creates evaluation successfully
- Researcher runs evaluation and gets results
- Researcher browses available benchmarks

**Validation Patterns:**

- **Command success/failure** - CLI exits with correct codes
- **Output format validation** - Progress displays, result summaries match expected structure
- **Data persistence** - Evaluations saved correctly, retrievable later
- **Response pattern matching** - LLM outputs contain reasoning structure (not exact content)

**Critical Path Testing:**

- Happy path through create → run → complete
- Authentication failures
- Invalid configurations
- API timeouts/errors
- Interrupted evaluations

**Test Categories:**

1. **Acceptance Tests** - End-to-end CLI scenarios
2. **Contract Tests** - OpenRouter API integration
3. **Repository Tests** - Database operations
4. **Domain Tests** - Business rule validation

**Non-deterministic Response Strategy:**

- Validate response structure/format rather than content
- Check reasoning traces contain expected patterns
- Verify extracted answers are properly formatted

Focus on research workflow validation rather than comprehensive coverage.

## See Also

- **[Core Behaviors](v2-core-behaviour-definition.md)** - User workflows requiring test validation
- **[Application Services Architecture](v2-application-services-architecture.md)** - Service coordination patterns to test
- **[Project Structure](v2-project-structure.md)** - Test organization and framework setup
- **[CLI Design](v2-cli-design.md)** - Command interface testing requirements
