# Neuro-Symbolic Mixture of Experts (NS-MoE)

This repository contains a collection of scripts and utilities designed to support the **Neuro-Symbolic Mixture of Experts (NS-MoE)** framework. The NS-MoE framework addresses the challenges of organizing and disseminating scholarly knowledge by integrating neural and symbolic models within a unified architecture. It enables the creation of **Cognitive Knowledge Graphs (CKGs)**, which enhance traditional Knowledge Graphs by incorporating contextual and multidisciplinary insights for structured organization of scholarly contributions.

---

## Contents

1. **[Data Generation and Refinement](#data-generation-and-refinement)**
    - See the [Detailed README for Data Generation](./src/generate_synthetic_data.md)
2. **[Training Symbolic Models](#training-symbolic-models)**
    - See the [Detailed README for Symbolic Models](./src/train_symbolic_model.md)
3. **[Fine-Tuning Models](#fine-tuning-models)**
    - See the [Detailed README for Fine-Tuning](./src/finetune.md)
4. **[Fine-Tuning NS-MoE](#fine-tuning-ns-moe)**
    - See the [Detailed README for NS-MoE Fine-Tuning](./src/generate_NSMoE_responses.md)
5. **[Mixture of Experts (MoE) Framework](#mixture-of-experts-moe-framework)**
    - See the [Detailed README for Mixture of Experts](./src/mixture_of_experts_framework.md)
6. **[Generating Model Responses](#generating-model-responses)**
    - See the [Detailed README for Generating Responses](./src/generate_model_responses.md)
7. **[Evaluation](#evaluation)**
    - See the [Detailed README for Evaluation](./src/evaluate.md)
8. **[Agent-Based Evaluation](#agent-based-evaluation)**
    - See the [Detailed README for Agent Evaluation](./src/agent_evaluation.md)

---

## Data Generation and Refinement
Generates synthetic data for research purposes, evaluates it based on specific criteria, and refines iteratively.

### Usage
```bash
python generate_synthetic_data.py \
  --base_dir <BASE_DIR> \
  --min_count <MIN_COUNT> \
  --iterations <ITERATIONS>
```

### Key Arguments
- `--base_dir`: Base directory for input and output files.
- `--min_count`: Minimum count of entries per field.
- `--iterations`: Number of refinement iterations.

**Environment Variable**: Set `OPENAI_API_KEY` for API integration.

For more details, see the [Data Generation README](./src/data_generation_refinement.md).

---

## Training Symbolic Models
Trains symbolic classifiers using TF-IDF for research tasks.

### Usage
```bash
python train_symbolic_model.py \
  --train_file <TRAIN_FILE> \
  --test_file <TEST_FILE> \
  --models_dir <MODELS_DIR> \
  --scores_dir <SCORES_DIR>
```

### Output
- Best model and vectorizer saved in `models_dir`.
- Classification scores saved in `scores_dir`.

For more details, see the [Symbolic Models README](./src/train_symbolic_model.md).

---

## Fine-Tuning Models
Fine-tune pre-trained transformer models using LoRA (Low-Rank Adaptation) for research tasks such as Predicate Label (PL) and Research Field (RF) prediction.

### Usage
```bash
python fine_tune_lora.py \
  --model_id <MODEL_ID> \
  --base_dir <BASE_DIR> \
  --data_dir <DATA_DIR>
```

### Key Arguments
- `--model_id`: Pre-trained model ID.
- `--base_dir`: Base directory for input/output storage.
- `--data_dir`: Directory containing the data for training.

For more details, see the [Fine-Tuning README](./src/finetune.md).

---

## Fine-Tuning NS-MoE
Fine-tunes models under the NS-MoE framework to create highly specialized experts for specific fields.

### Key Features
- Incorporates **dynamic routing** mechanisms.
- Efficiently handles tasks across multiple fields with minimal computational resources.

### Usage
Refer to the [NS-MoE Fine-Tuning README](./src/generate_NSMoE_responses.md) for detailed instructions.

---

## Mixture of Experts (MoE) Framework
Implements an MoE system for routing tasks to specialized models and generating field-specific responses.

### Usage
```bash
python generate_NSMoE_responses.py \
  --model_id <MODEL_ID> \
  --base_dir <BASE_DIR> \
  --version <VERSION>
```

For more details, see the [Mixture of Experts Framework README](./src/mixture_of_experts_framework.md).

---

## Generating Model Responses
Supports generating predictions for tasks like Predicate Label (PL) and Research Field (RF) using fine-tuned or base transformer models.

### Usage
```bash
python generate_model_response.py \
  --model_type <MODEL_TYPE> \
  --task <TASK_TYPE> \
  --model_id <MODEL_ID> \
  --data_file <DATA_FILE> \
  --output_dir <OUTPUT_DIR>
```

For more details, see the [Model Response README](./src/generate_model_responses.md).

---

## Evaluation
Evaluate model predictions for PL and RF tasks based on accuracy, precision, recall, and F1-score.

### Usage
```bash
python evaluate.py \
  --model_type <MODEL_TYPE> \
  --model_ids <MODEL_IDS> \
  --prompt_types <PROMPT_TYPES> \
  --base_dir <BASE_DIR> \
  --threshold <THRESHOLD>
```

For more details, see the [Evaluation README](./src/evaluate.md).

---

## Agent-Based Evaluation
Leverages GPT-4o for evaluating model outputs in tasks like predicate label prediction.

### Usage
```bash
python agent_evaluation.py \
  --model_type <MODEL_TYPE> \
  --model_id <MODEL_ID> \
  --base_dir <BASE_DIR>
```

For more details, see the [Agent Evaluation README](./src/agent_evaluation.md).

---

## Requirements

Install the required Python libraries:
```bash
pip install torch transformers pandas tqdm openai datasets peft bitsandbytes
```
