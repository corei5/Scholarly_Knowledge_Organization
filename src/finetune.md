# Fine-Tune Models with LoRA

The `finetune.py` script is designed to fine-tune pre-trained transformer models using LoRA (Low-Rank Adaptation) for tasks such as Research Field Classification (RF) and Predicate Label Prediction (PL).

---

## Requirements

### Python Libraries
- `torch`
- `transformers`
- `pandas`
- `datasets`
- `trl`
- `bitsandbytes`
- `peft`
- `huggingface_hub`

### Installation
Install the required Python libraries:
```bash
pip install torch transformers pandas datasets trl bitsandbytes peft huggingface_hub
```

---

## Usage

Run the script from the command line, specifying the necessary arguments:

```bash
python finetune.py \
  --model_id <MODEL_ID> \
  --task <TASK_TYPE> \
  --base_dir <BASE_DIR> \
  --data_dir <DATA_DIR> \
  --huggingface_token <HUGGINGFACE_TOKEN> \
  --epochs <EPOCHS> \
  --batch_size <BATCH_SIZE> \
  --learning_rate <LEARNING_RATE> \
  --train_file <TRAIN_FILE> \
  --test_file <TEST_FILE>
```

### Arguments
- `--model_id` (required): The ID of the pre-trained model to be fine-tuned (e.g., `meta-llama/Llama-3.1-8B`).
- `--task` (required): The task type. Options:
  - `rf` for Research Field Classification.
  - `pl` for Predicate Label Prediction.
- `--base_dir` (required): Base directory for saving model outputs.
- `--data_dir` (required): Directory containing data files.
- `--huggingface_token` (required): Hugging Face authentication token.
- `--epochs` (optional): Number of training epochs (default: `2`).
- `--batch_size` (optional): Batch size for training and evaluation (default: `1`).
- `--learning_rate` (optional): Learning rate for training (default: `2e-4`).
- `--train_file` (required): Path to the training dataset file (Pickle format).
- `--test_file` (required): Path to the testing dataset file (Pickle format).

### Example Command
```bash
python finetune.py \
  --model_id meta-llama/Llama-3.1-8B \
  --task rf \
  --base_dir /path/to/base_dir \
  --data_dir /path/to/data_dir \
  --huggingface_token your_huggingface_token \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 3e-4 \
  --train_file /path/to/train.pkl \
  --test_file /path/to/test.pkl
```

---

## Output
- Fine-tuned models are saved in the specified `base_dir` under a folder named after the task and model type.
- Example structure:
  ```
  /path/to/base_dir/models_rf/
      ├── llama-3.1-8B_RF
      │   ├── config.json
      │   ├── pytorch_model.bin
      │   └── ...
  ```
- Logs provide training and evaluation details.

---

## Notes
- **Training Data Format**: The training and testing data should be provided in Pickle (`.pkl`) format, with the following columns:
  - `title`
  - `abstract`
  - `Field` (for Research Field Classification)
  - `predicate_label` (for Predicate Label Prediction)

- **Task-Specific Prompts**:
  - For RF:
    ```
    Based on the provided information, determine the most appropriate research field.
    Paper Title: <TITLE>
    Abstract: <ABSTRACT>
    ```
  - For PL:
    ```
    Based on the provided information, determine the most appropriate Predicate Labels.
    Paper Title: <TITLE>
    Abstract: <ABSTRACT>
    ```

- **Environment Variables**: Ensure the Hugging Face token is set up as an argument (`--huggingface_token`).
