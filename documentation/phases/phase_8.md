# Phase 8: Basic Testing for Output Parsing & Results Processing

## Purpose

Simple functional tests to ensure Phase 6 (parsing) and Phase 7 (database) work correctly for researchers.

**Timeline**: 4 hours total

## Testing Approach

### Test 1: Parsing Works (1.5 hours)

```python
def test_instructor_parsing():
    """Test structured parsing extracts answers correctly"""
    sample_output = "Step 1: Calculate... Step 2: The answer is 42."
    result = instructor_parser.extract(sample_output)
    assert result.final_answer == "42"

def test_parsing_fallback():
    """Test regex fallback when structured parsing fails"""
    # Mock instructor failure, ensure regex still works

def test_all_reasoning_approaches():
    """Test parsing works with each reasoning approach"""
    for approach in ["CoT", "PoT", "Reflection", etc.]:
        # Basic smoke test with sample output
```

### Test 2: Database Works (1.5 hours)

```python
def test_save_and_retrieve():
    """Test saving experiment results to database"""
    result = StandardResponse(...)
    db.save_run_result(result)
    retrieved = db.get_run(result.id)
    assert retrieved.parsed_answer == result.parsed_answer

def test_export_formats():
    """Test CSV/JSON/Excel exports work"""
    db.export_to_csv(experiment_id, "test.csv")
    assert os.path.exists("test.csv")
```

### Test 3: Integration Works (1 hour)

```python
def test_end_to_end():
    """Test complete flow: run experiment → parse → save → export"""
    config = ExperimentConfig(sample_count=3, approaches=["CoT"])
    runner = ExperimentRunner(config)
    runner.run_single_experiment()

    # Check results saved to database
    results = db.get_experiment_results(config.experiment_id)
    assert len(results) == 3
    assert all(r.parsed_answer is not None for r in results)
```

## Success Criteria

- Parsing extracts answers from sample outputs
- Database saves and retrieves data correctly
- Exports generate valid files
- End-to-end flow completes without errors

That's it. Simple functional validation for a research utility.
