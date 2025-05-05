# Evaluate Models for Predicate Label and Research Field Tasks

This script evaluates models for tasks such as predicate label prediction and research field classification. It calculates accuracy, precision, recall, and F1-score metrics for model predictions based on similarity thresholds.

---

## Requirements

### Python Libraries

- `pandas`
- `sentence-transformers`
- `tqdm`
- `argparse`

### Installation

Install the required Python libraries using pip:

```bash
pip install pandas sentence-transformers tqdm
```

---

## Usage

Run the script from the command line, providing the required arguments:

```bash
python evaluate.py --model_type <MODEL_TYPE> \
                          --model_ids <MODEL_ID_1> <MODEL_ID_2> ... \
                          --prompt_types <PROMPT_TYPE_1> <PROMPT_TYPE_2> ... \
                          --base_dir <BASE_DIR> \
                          --threshold <THRESHOLD> \
                          --version <VERSION>
```

### Arguments

- `--model_type` (required): The type of model to evaluate. Options: `MoE`, `FT`, `Base`.
- `--model_ids` (required): Space-separated list of model IDs to evaluate. Example: `meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2`.
- `--prompt_types` (required): Space-separated list of prompt types. Example: `MoE_predicate_label MoE_research_field`.
- `--base_dir` (required): Base directory for model outputs and scores.
- `--threshold` (optional): Cosine similarity threshold for evaluation. Default: `0.90`.
- `--version` (optional): Version identifier for evaluation. Default: `v1`.

### Example Command

Evaluate a Mixture of Experts (MoE) model:

```bash
python evaluate.py --model_type MoE \
                          --model_ids meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2 \
                          --prompt_types MoE_predicate_label MoE_research_field \
                          --base_dir /nfs/home/zamilp/SKO \
                          --threshold 0.90 \
                          --version v4
```

---

## Output

- The script saves evaluation results as pickle files in the `scores` directory under the specified `base_dir`.
- Example structure:
  ```
  /nfs/home/user/SKO/scores/
      └── MoE_v4/
          ├── predicate_label_scores.pkl
          └── research_field_scores.pkl
  ```

---

## Key Functions

### `calculate_accuracy_pl`

Calculates row-wise accuracy for predicate labels based on cosine similarity.

### `calculate_metrics_pl`

Calculates precision, recall, and F1-score for predicate labels.

### `calculate_accuracy_rf`

Evaluates the accuracy of single research field predictions.

### `evaluate_models`

Main function to evaluate the models, process data, and save results.

---

## Notes

- Ensure that the model outputs are stored in the directory specified by `base_dir` and follow the required structure.
- Customize the script as needed to use other SentenceTransformers models by replacing `'Lajavaness/bilingual-embedding-large'` with the desired model.
