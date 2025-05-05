## Overview

The `generate_ft_model_responses.py` script generates responses for **Research Field (RF)** and **Predicate Label (PL)** tasks using fine-tuned transformer models. The script dynamically loads models, prepares prompts, and generates predictions for each task. It supports saving results for further analysis.

---

## Features

- **Task Support**:
  - **Research Field (RF)**: Predicts the most appropriate research field for a paper.
  - **Predicate Label (PL)**: Generates predicate labels in JSON format for a research paper.
- **Dynamic Prompt Creation**:
  - Automatically generates prompts tailored to each task.
- **Efficient Response Generation**:
  - Uses the Hugging Face `pipeline` for optimized text generation.
- **Output Management**:
  - Saves responses and response times in a structured format for further evaluation.

---

## Requirements

### Python Libraries

- `torch`
- `transformers`
- `pandas`
- `tqdm`

### Installation

Install the required Python libraries using:
```bash
pip install torch transformers pandas tqdm
```

---

## Usage

Run the script using the following command:

```bash
python generate_ft_model_responses.py \
  --base_dir <BASE_DIR> \
  --task <TASK_TYPE> \
  --model_name <MODEL_NAME> \
  --output_dir <OUTPUT_DIR> \
  --max_tokens <MAX_TOKENS>
```

### Arguments

- `--base_dir` (required): The base directory containing model checkpoints and data files.
- `--task` (required): Task type:
  - `rf`: Research Field.
  - `pl`: Predicate Label.
- `--model_name` (required): Name of the fine-tuned model to use.
- `--output_dir` (required): Directory to save the generated responses.
- `--max_tokens` (optional): Maximum number of tokens to generate per response (default: `512`).

---

## Input Data Format

The input test data must be stored as a Pickle file (`.pkl`) with the following columns:
- `title`: The title of the research paper.
- `abstract`: The abstract of the research paper.

Default test data file: `SKO_with_taxonomy.pkl` (located in the `base_dir`).

---

## Output

The script generates and saves responses in a Pickle file with the following structure:
- `responses`: A list of tuples containing:
  - `title`: The title of the research paper.
  - `raw_response`: The raw output from the model.
- `times`: A list of response times for each input.

### Output File Naming

Output files are saved in the format:
```
<OUTPUT_DIR>/<MODEL_NAME>_<TASK_TYPE>_responses.pkl
```

---

## Example Command

```bash
python generate_ft_model_responses.py \
  --base_dir /path/to/base_dir \
  --task pl \
  --model_name fine_tuned_model \
  --output_dir /path/to/output_dir \
  --max_tokens 512
```

---

## Notes

- Ensure the `base_dir` contains the required model checkpoint (`best_checkpoints.pkl`) and test data (`SKO_with_taxonomy.pkl`).
- Adjust the `max_tokens` parameter based on your task and resource constraints.
- Generated outputs can be analyzed or used for downstream evaluations.
