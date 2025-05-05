## Requirements

### Python Libraries
Install the required packages using `pip`:
```bash
pip install transformers peft trl torch pandas datasets bitsandbytes
```

### GPU Support
This script is optimized for systems with CUDA-enabled GPUs for faster training and evaluation.

---

## Usage

### Command-Line Arguments
The script accepts the following arguments:
- `--model_id`: (Required) The Hugging Face model ID to fine-tune (e.g., `meta-llama/Llama-2-7b-hf`).
- `--base_dir`: (Required) Base directory for storing models and data.
- `--data_dir`: (Required) Directory containing the training and generated data.

### Example Command
```bash
python fine_tune_lora.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --base_dir /path/to/project/base \
  --data_dir /path/to/data
```

---

## Input Data Structure

1. **Base Dataset** (`SKO_with_taxonomy.pkl`):
   - Contains fields like `title`, `abstract`, and `Field` for task-specific data.

2. **Generated Data** (`generated_data_v3.pkl`):
   - Augmented data with fields like `predicate_label`.

Both files must be stored in the specified `base_dir` and `data_dir`.

---

## Output

1. **Trained Models**:
   - Saved in `base_dir/models_MoE` with names following the format `<model_prefix>_PL_<field_name>`.

2. **Logs**:
   - Detailed logs are written to `fine_tuning.log`.

---

## Workflow

1. **Load Datasets**:
   - Reads `SKO_with_taxonomy.pkl` (base dataset) and `generated_data_v3.pkl` (generated data).

2. **Field Detection**:
   - Automatically detects unique fields from the dataset.

3. **Training**:
   - Trains the model for each field independently using LoRA and QLoRA configurations.

4. **Evaluation**:
   - Evaluates the trained model on field-specific test data.
   - Logs evaluation results, including loss, for each field.

5. **Resource Cleanup**:
   - Deletes the model and trainer after each field is processed.
   - Runs garbage collection and clears GPU memory.

---

## Configuration Details

### LoRA Configuration
- `r`: 16
- `lora_alpha`: 32
- `lora_dropout`: 0.05
- `task_type`: `CAUSAL_LM`

### QLoRA Configuration
- `bnb_4bit_quant_type`: `nf4`
- `bnb_4bit_compute_dtype`: `torch.float16`
- `bnb_4bit_use_double_quant`: `True`

---

## Key Functions

1. **`find_all_linear_names`**:
   - Identifies all linear layers in the model for applying LoRA.

2. **`format_chat_template`**:
   - Prepares rows for fine-tuning by creating a chat-based input structure.

3. **Field-Specific Training**:
   - Trains a separate model for each field and saves the results.

4. **Dynamic Resource Management**:
   - Ensures memory is cleared after each training session.