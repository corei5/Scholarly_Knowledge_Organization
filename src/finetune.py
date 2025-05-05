import logging
import os
import torch
import pandas as pd
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from trl import SFTTrainer
import gc
import argparse

# Configure logging to console only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
    ]
)

logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune transformer models using LoRA")
parser.add_argument('--model_id', type=str, required=True, help='The ID of the pre-trained model to be fine-tuned')
parser.add_argument('--task', type=str, required=True, choices=['rf', 'pl'], help='Task type: "rf" for research field, "pl" for predicate label')
parser.add_argument('--base_dir', type=str, required=True, help='Base directory for data and output files')
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing input data files')
parser.add_argument('--huggingface_token', type=str, required=True, help='Hugging Face authentication token')
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs (default: 2)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and evaluation (default: 1)')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for training (default: 2e-4)')
parser.add_argument('--train_file', type=str, required=True, help='Path to the training dataset file')
parser.add_argument('--test_file', type=str, required=True, help='Path to the testing dataset file')
args = parser.parse_args()

# Hugging Face Login
login(token=args.huggingface_token)

# QLoRA config for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Function to format rows for the model
def format_chat_template(row, task):
    if task == 'rf':
        row_json = [
            {
                "role": "user", "content": f"""Based on the provided information, determine the most appropriate research field.\nPaper Title: {row['title']}\nAbstract: {row['abstract']}"""
            },
            {
                "role": "assistant", "content": f"""research_field: "{row['Field']}"""
            }
        ]
    elif task == 'pl':
        row_json = [
            {
                "role": "user", "content": f"""Based on the provided information, determine the most appropriate Predicate Labels.\nPaper Title: {row['title']}\nAbstract: {row['abstract']}"""
            },
            {
                "role": "assistant", "content": f"""predicate_labels: "{row['predicate_label']}"""
            }
        ]
    row["formatted_text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Load data
BASE_DIR = args.base_dir
DATA_DIR = args.data_dir
MODEL_DIR = f'{BASE_DIR}/models_{args.task}'

# Create the model directory if it does not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load dataframes from user-provided paths
logger.info("Loading dataframes...")
train_data = pd.read_pickle(args.train_file)
test_data = pd.read_pickle(args.test_file)
logger.info("Dataframes loaded successfully.")

# Use the model_id from script arguments
model_id = args.model_id

task_type = args.task
logger.info("=" * 50)
logger.info(f"Starting fine-tuning for model: {model_id} on task: {task_type}...")
base_model = model_id
new_model_prefix = base_model.split('/')[1] + f'_{task_type.upper()}'
new_model_prefix = os.path.join(MODEL_DIR, new_model_prefix)

# Load tokenizer once
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
logger.info("Tokenizer loaded successfully.")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data[['title', 'abstract', 'Field', 'predicate_label']])
test_dataset = Dataset.from_pandas(test_data[['title', 'abstract', 'Field', 'predicate_label']])
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

dataset = dataset.map(lambda row: format_chat_template(row, task_type), num_proc=4)

# Reload base model and apply LoRA configuration
logger.info(f"Loading {base_model}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA config
modules = find_all_linear_names(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)

model = get_peft_model(model, peft_config)
logger.info(f"Base model and LoRA configuration loaded.")

training_arguments = TrainingArguments(
    output_dir=new_model_prefix,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=args.epochs,
    eval_strategy='steps',
    eval_steps=500,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=args.learning_rate,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
)

model.config.use_cache = False
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    dataset_text_field="formatted_text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments
)
# Disable cache during training to avoid conflict
model.config.use_cache = False

# Train the model
logger.info(f"Starting training...")
trainer.train()
logger.info(f"Training completed.")

# Evaluate the model
logger.info(f"Starting evaluation...")
eval_results = trainer.evaluate()
current_loss = eval_results['eval_loss']
logger.info(f"Evaluation completed. Loss: {current_loss}\n")
