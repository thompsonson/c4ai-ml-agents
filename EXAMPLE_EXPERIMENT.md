# Example Experiment: BBEH Logical Reasoning Analysis

This document provides step-by-step instructions for running your first reasoning experiment using the Cohere Labs ML Agents framework. This experiment tests when different reasoning approaches help with hard logical reasoning problems.

## Experiment Overview

**Goal**: Compare reasoning approaches on logical reasoning problems to understand when and how reasoning improves AI performance.

**Dataset**: BBEH (Beliefs, Biases, Emotions, and Heuristics) - Hard logical reasoning tasks
**Source**: `MrLight/bbeh-eval` (Row #17 in [Cohere Labs Benchmarks](https://docs.google.com/spreadsheets/d/1-TBlPSIiBymQfCdF_LCYJznLwaxcKtYTRJME0NT17kU/edit?gid=389132052#gid=389132052))
**Task Type**: Generation (open-ended answers)
**Difficulty**: HARD
**Official Sample Count**: 2,000 (we'll start with 50 for testing)

## Why This Dataset?

The BBEH dataset is perfect for reasoning research because:
- Contains logical puzzles that require multi-step thinking
- Hard difficulty ensures reasoning approaches have room to improve
- Already curated by the Cohere Labs community
- Has clear right/wrong answers for evaluation
- Tests cognitive biases and logical fallacies

## Prerequisites

1. ✅ Environment setup completed (see [README.md](./README.md))
2. ✅ Virtual environment activated: `source .venv/bin/activate`
3. ✅ API keys configured in `.env` file
4. ✅ Jupyter notebook running: `jupyter notebook`

## Step 1: Configure API Keys

Add your OpenRouter API key to `.env`:
```bash
# Get a free key from https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**💡 Tip**: We'll use the FREE `openai/gpt-oss-20b:free` model to minimize costs.

## Step 2: Open and Prepare the Notebook

1. Open `Reasoning_LLM.ipynb` in Jupyter
2. Run the first cell (Setup) - this installs all dependencies
3. Skip the API key input prompts (we'll use the .env file)

## Step 3: Configure the Experiment

In **Cell 4** (Experiment Configuration), set these values:

| Setting | Value | Reasoning |
|---------|-------|-----------|
| **Provider** | `openrouter` | Access to free models |
| **Model** | `openai/gpt-oss-20b:free` | Free tier, good performance |
| **Temperature** | `0.3` | Consistent results, some creativity |
| **Max Tokens** | `512` | Enough for reasoning, cost-effective |
| **Top P** | `0.9` | Standard setting |
| **Reasoning** | `Chain-of-Thought (CoT)` | Start with most common approach |

## Step 4: Limit Dataset Size (Testing)

In **Cell 6** (Load Dataset), modify the code:

```python
# After loading the dataset, add this line:
hf_dataset = hf_dataset[0:50]  # Use first 50 samples for testing
print(f"Using {len(hf_dataset)} samples for testing")
```

**Why 50 samples?**: Quick testing (~5-10 minutes runtime) before committing to full 2,000 sample run.

## Step 5: Run Your First Experiment

1. **Execute Cell 8**: Core inference functions (no changes needed)
2. **Execute Cell 10**: Run the experiment
   - Watch the progress bar
   - Should complete in 5-10 minutes
   - Results will display in a table

## Step 6: Analyze Results

Look for these patterns in your results:

### Key Metrics to Track:
- **Execution Time**: How long did reasoning take per problem?
- **Model Output Quality**: Do the answers look more detailed/structured?
- **Reasoning Traces**: Check the full reasoning process in the trace column

### Expected Patterns:
- **Longer responses** with Chain-of-Thought
- **Step-by-step breakdowns** in the model output
- **Higher execution times** (~3-5x longer than direct answers)

## Step 7: Compare with Baseline

Run a second experiment with **Reasoning = "None"** to compare:

1. Change reasoning approach to "None"
2. Run Cell 10 again
3. Compare the two result tables

### Analysis Questions:
- Are the Chain-of-Thought answers more detailed?
- Do they show step-by-step reasoning?
- Which approach feels more accurate to you?

## Step 8: Save and Share Results

1. **Download CSV**: The notebook automatically saves results
2. **File format**: `gpt-oss-20b_free_Chain-of-Thought(CoT)_YYYYMMDD-HHMMSS.csv`
3. **Share**: Upload to Discord #ml-agents channel with brief summary

## Example Results Summary

```markdown
**BBEH Logical Reasoning - Initial Results**

- Dataset: 50 samples from MrLight/bbeh-eval
- Model: openai/gpt-oss-20b:free
- Approaches tested: None vs Chain-of-Thought

Key findings:
- CoT responses 3x longer on average
- CoT showed explicit step-by-step reasoning
- Execution time: ~8 seconds per problem with CoT vs ~2 seconds direct
- [Attach CSV files]

Next: Planning to test Tree-of-Thought and full 2000 sample run.
```

## Next Steps: Full Experiment

Once you're comfortable with the process:

### Phase 2: Multiple Reasoning Approaches
Test these on the same 50 samples:
- None (baseline)
- Chain-of-Thought
- Tree-of-Thought
- Program-of-Thought (if problems involve math)

### Phase 3: Full Dataset
- Increase to full 2,000 samples
- Run overnight for complete results
- Focus on your best-performing approach

### Phase 4: Choose New Dataset
Pick from Cohere Labs benchmarks:
- **MATH [Easy]**: Math problems (`ck46/hendrycks_math`)
- **StrategyQA**: Common sense reasoning (`ChilleD/StrategyQA`)
- **MBPP**: Python coding (`Muennighoff/mbpp`)

## Troubleshooting

### Common Issues:

**Error: "API key not found"**
- Check your `.env` file has the correct key
- Restart Jupyter after editing `.env`

**Error: "Rate limit exceeded"**
- Free models have usage limits
- Wait 5-10 minutes and retry
- Consider using a paid model for larger runs

**Slow performance**
- Normal for reasoning approaches (3-10x slower)
- Use smaller sample sizes for testing
- Run overnight for full experiments

**Unexpected results**
- Check the reasoning trace column for debugging
- Verify your prompt format
- Try different temperature settings (0.1-0.7)

## Cost Estimation

| Model Type | 50 samples | 2000 samples | Notes |
|------------|------------|--------------|-------|
| **Free models** | $0 | $0 | Rate limited |
| **GPT-5-mini** | ~$0.10 | ~$4 | Fast, reliable |
| **Claude Sonnet** | ~$0.50 | ~$20 | High quality |

## Contributing to the Community

Your results help answer the core research questions:
1. **When does reasoning help?** (baseline vs reasoning comparison)
2. **Which reasoning works best?** (approach comparison)
3. **What's the cost-benefit?** (time/cost vs quality trade-off)

Share your findings in Discord and consider:
- Submitting interesting examples
- Proposing new reasoning approaches
- Testing additional datasets
- Contributing code improvements

## Resources

- [Cohere Labs ML Agents Discord](https://discord.gg/ckaQnUakYx)
- [Project Benchmarks Spreadsheet](https://docs.google.com/spreadsheets/d/1-TBlPSIiBymQfCdF_LCYJznLwaxcKtYTRJME0NT17kU/edit)
- [OpenRouter Free Models](https://openrouter.ai/models?order=newest&supported_parameters=tools&pricing=free)
- [Reasoning Techniques Overview](https://github.com/antropics/reasoning-techniques)
