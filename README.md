# Bengali Poem Generator

## How to use

You need the UV package/project manager to install the dependencies.  
You can get it from [here](https://docs.astral.sh/uv/getting-started/installation/).

Set up the environment. (Only once)

```bash
uv venv
.venv/Scripts/activate
# uv pip install torch --index-url https://download.pytorch.org/whl/cu118 --link-mode=symlink
uv sync --link-mode=symlink --extra=cpu   # for CPU only
uv sync --link-mode=symlink --extra=cu124 # for CUDA support
```

To run any script, append `uv run` before the `python` command.

## Gated Repos on HuggingFace

Some files like [bengali-poem-generator/utils/compare_tokenizers.py](./bengali-poem-generator/utils/compare_tokenizers.py),
require access Gated Public Repositories on HuggingFace.

You must log in to HuggingFace using `huggingface-cli`.

1. [Generate a token with the `Read access to contents of all public gated repos you can access` permission](https://huggingface.co/settings/tokens/new?tokenType=fineGrained) on HuggingFace. (Account Required)
2. Run `huggingface-cli login` and paste the token generated in the previous step.


1. How to Run the Code for Fine-tuning:
1.1) Sarvam AI:
Make a copy of this notebook and run it on Google Colab
https://colab.research.google.com/drive/1tpH_2PEIj7S84OpeYEUFtWisEQYKK6zt?usp=sharing

1.2) Gemma-3 1b:
Make a copy of this notebook and run it on Google Colab
https://colab.research.google.com/drive/1c8eMf8-I_udIgQtrULKo-HUUYB8BVQLx?usp=sharing

1.3) Gemma-3 4b:
Make a copy of this notebook and run it on Google Colab
https://colab.research.google.com/drive/1gGLcXnX49dCvHgCw42E_EluX4uJLCouo?usp=sharing


# Bengali Poem Dataset Processing and Evaluation

This repository contains a collection of Python scripts designed to process the Bengali Poem Dataset, split it into training/testing sets, and evaluate a language model’s performance on generating Bengali poetry. The scripts handle data preprocessing, dataset splitting, model inference, and metric computation.

## Overview

The scripts work together to:
1. Process raw poem data from the Bengali Poem Dataset.
2. Split the dataset into training, validation, and testing sets.
3. Evaluate a pre-trained language model on the test set using metrics like BLEU, ROUGE, and ChRF.
4. Analyze tokenizers for various language models.

All scripts assume a `data` folder in the same directory for storing input and output files.

## Prerequisites

- **Python 3.8+**
- **Dependencies**: Install required packages using:
  ```bash
  pip install torch transformers datasets evaluate tqdm
  ```
- **Bengali Poem Dataset**: Clone the dataset using:
  ```bash
  cd ..
  git clone https://github.com/shuhanmirza/Bengali-Poem-Dataset
  ```
- **Pre-trained Model**: A model checkpoint (e.g., from SARVAM) stored at `D:/College/M.Tech/INLP/SARVAM` (update path as needed).
- **Disk Space**: Ensure sufficient space for JSON/JSONL files in the `data` folder.

## Folder Structure

```
.
├── data/                  # Folder for input/output data files
├── make_datasets.py       # Processes raw poem data
├── evaluate_model.py      # Evaluates model performance
├── split_dataset.py       # Splits dataset into train/test
├── merge_and_split.py     # Merges and splits into train/val/test
├── tokenizer_analysis.py  # Analyzes tokenizer properties
└── README.md              # This file
```

## Usage

### 1. `make_datasets.py`
**Purpose**: Processes the Bengali Poem Dataset and creates two JSON files: one for poems and one for thematic instructions.

**Run**:
```bash
python make_datasets.py
```

**Output**:
- `data/poems.json`: Poems with random line-based instructions.
- `data/classes.json`: Poems with thematic instructions.

**Notes**: Assumes the dataset is cloned at `../Bengali-Poem-Dataset`.

### 2. `split_dataset.py`
**Purpose**: Splits `data/poems.json` into training and testing sets.

**Run**:
```bash
python split_dataset.py
```

**Output**:
- `data/train.json`: 80% of the data.
- `data/test.json`: 20% of the data.

### 3. `merge_and_split.py`
**Purpose**: Merges `poems.json` and `classes.json`, then splits into train/validation/test sets in JSONL format.

**Run**:
```bash
python merge_and_split.py
```

**Output**:
- `data/train.jsonl`: 80% of merged data.
- `data/val.jsonl`: 10% of merged data.
- `data/test.jsonl`: 10% of merged data.

### 4. `evaluate_model.py`
**Purpose**: Evaluates a pre-trained model on `data/test.jsonl` using BLEU, ROUGE, and ChRF metrics.

**Run**:
```bash
python evaluate_model.py
```

**Output**:
- `out/combined_metrics_results.txt`: Metric scores.

**Notes**: Update `MODEL_DIR` in the script to your model path.

### 5. `tokenizer_analysis.py`
**Purpose**: Analyzes tokenizers from various pre-trained models on a sample Bengali text.

**Run**:
```bash
python tokenizer_analysis.py
```

**Output**: Prints tokenizer details (vocab size, token IDs, etc.) to the console.

## Notes
- Ensure the `data` folder exists or is created by the scripts.
- Modify file paths (e.g., `MODEL_DIR`, dataset location) as per your setup.
- Scripts use UTF-8 encoding for Bengali text compatibility.

### Explanation
- **Overview**: Briefly describes the purpose of the scripts.
- **Prerequisites**: Lists dependencies and setup steps.
- **Folder Structure**: Reflects the flat structure with a `data` folder inferred from the scripts.
- **Usage**: Provides instructions for running each script with inputs/outputs.
- **Notes**: Highlights customization needs (e.g., paths).
