# README: Mixture of Experts (MoE) Framework

## Overview
This script implements a Mixture of Experts (MoE) framework to classify and route tasks to field-specific fine-tuned language models using a gatekeeper model. It processes text data, predicts fields, generates responses, and extracts predicate labels.

## Features
- **Gatekeeper Model**: Routes tasks to appropriate field-specific models using SVM.
- **Fine-Tuned Models**: Dynamically loads fine-tuned models for specific fields.
- **Predicate Label Extraction**: Extracts and processes structured predicate labels from model responses.
- **Progress Saving**: Saves intermediate results to handle long-running tasks and errors.

## Usage
Run the script with the following arguments:
```bash
python generate_NSMoE_responses.py --model_id <model_id> --base_dir <base_directory> --version <version>
```
- `--model_id`: Hugging Face model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
- `--base_dir`: Base directory for data and models.
- `--version`: Version of the MoE framework to use.

## Output
Processed responses are saved as a pickle file in the specified output directory.
