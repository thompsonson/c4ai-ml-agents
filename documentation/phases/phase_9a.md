# Phase 9a: Dataset Upload to HuggingFace Hub

## Purpose

Upload processed datasets to `c4ai-ml-agents/<dataset_name>` with automated metadata generation.

**Timeline**: 2-3 hours

## Implementation

### CLI Command

```bash
# Upload processed dataset
ml-agents preprocess-upload spatial_eval_processed.csv \
  --source-dataset MilaWang/SpatialEval \
  --target-name SpatialEval \
  --config tqa \
  --description "Processed SpatialEval dataset in standardized INPUT/OUTPUT format"

# Generic pattern: c4ai-ml-agents/<target-name>
```

### Core Functionality

```python
class DatasetUploader:
    def __init__(self, org_name: str = "c4ai-ml-agents"):
        self.org_name = org_name

    def upload_dataset(self,
                      processed_file: str,
                      source_dataset: str,
                      target_name: str,
                      config: str = None,
                      description: str = None) -> str:
        """Upload processed dataset to HF Hub"""

        # Load processed data
        dataset = Dataset.from_csv(processed_file)

        # Generate metadata
        card_content = self._generate_dataset_card(
            source_dataset, config, description
        )

        # Upload to hub
        repo_id = f"{self.org_name}/{target_name}"
        dataset.push_to_hub(repo_id)

        # Upload dataset card
        self._upload_dataset_card(repo_id, card_content)

        return repo_id
```

### Auto-Generated Metadata

```python
def _generate_dataset_card(self, source: str, config: str, desc: str) -> str:
    return f"""---
dataset_info:
  features:
  - name: INPUT
    dtype: string
  - name: OUTPUT
    dtype: string
  splits:
  - name: train
    num_examples: {{num_examples}}
task_categories:
- question-answering
source_datasets:
- {source}
---

# {target_name}

Processed version of [{source}](https://huggingface.co/datasets/{source})
in standardized INPUT/OUTPUT format for ML Agents reasoning evaluation.

**Config used**: {config}
**Processing**: {desc}

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("c4ai-ml-agents/{target_name}")
```

"""

```

### Authentication
```python
# Requires HF token with write access to c4ai-ml-agents
def authenticate():
    token = os.getenv("HF_TOKEN") or input("Enter HF token: ")
    login(token=token)
```

## Success Criteria

- Processed datasets upload successfully to `c4ai-ml-agents/<name>`
- Generated dataset cards include source attribution and usage info
- CLI provides upload progress and final hub URL
- Error handling for authentication and upload failures
- All documentation is updated (README, benchmark CSV, roadmap)
