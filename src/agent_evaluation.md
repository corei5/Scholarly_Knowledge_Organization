# Agent Evaluation

This script evaluates model outputs using GPT-4o agent for tasks like predicate label prediction and research field classification.

## Requirements

### Python Libraries
- `pandas`
- `openai`
- `tqdm`
- `argparse`
- `logging`

### Installation
Install the required Python libraries using pip:
```bash
pip install pandas openai tqdm
```

---

## Usage

Run the script from the command line, passing the necessary arguments:

```bash
python agent_evaluation.py \
  --model_type <MODEL_TYPE> \
  --model_id <MODEL_ID> \
  --base_dir <BASE_DIR> \
  --version <VERSION>
```

### Arguments
- `--model_type` (required): Specifies the type of model being evaluated. Options:
  - `MoE` (Mixture of Experts)
  - `FT` (Fine-Tuned)
  - `Base` (Base Models)
- `--model_id` (required): The ID of the model being evaluated (e.g., from Hugging Face).
- `--base_dir` (required): Directory where input files and results are stored.
- `--version` (optional): Version identifier for the evaluation. Default is `v1`.

---

## Example Command

Evaluate a fine-tuned model:
```bash
python agent_evaluation.py \
  --model_type FT \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --base_dir /nfs/home/zamilp/SKO \
  --version v2
```

---

## Environment Variables
The API key for GPT-4o must be set as an environment variable:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
Ensure this environment variable is configured before running the script.

---

## Output
The script generates the following outputs:
- Evaluation results saved as pickle files in the specified `base_dir`:
  ```
  <BASE_DIR>/output/<MODEL_TYPE>_<VERSION>/
      ├── <MODEL_NAME>_gpt-4o_evaluated.pkl
      └── <MODEL_NAME>.pkl
  ```
- Logs provide real-time updates on evaluation progress and errors.

---

## Key Functions

### `create_predicate_label_evaluation_prompt`
Generates prompts for evaluating predicate labels.

### `create_research_field_evaluation_prompt`
Generates prompts for evaluating research field classifications.

### `evaluate_response`
Sends prompts to the GPT-4o API and retrieves evaluation responses.

### `evaluate_model_responses`
Handles parallel processing for evaluating multiple rows of data.

---

## Notes
- Ensure the input files (`<MODEL_NAME>.pkl`) are correctly formatted and stored in the specified `base_dir`.
- Use the `--version` argument to manage multiple evaluations of the same model.

---

## License
This script is provided for evaluation purposes. Modify and use as needed.

