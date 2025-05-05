import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
)
import os
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from trl import SFTTrainer
import gc
import argparse

# Parse script parameters
parser = argparse.ArgumentParser(description="Fine-tune a transformer model with LoRA and QLoRA.")
parser.add_argument("--model_id", type=str, required=True, help="The ID of the model to fine-tune.")
parser.add_argument("--base_dir", type=str, required=True, help="Base directory for data and model storage.")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data.")
args = parser.parse_args()

# Configure logging to both console and file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
def format_chat_template(row):
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

# Load dataframes from pickled files
logger.info("Loading dataframes...")
base_df = pd.read_pickle(f'{BASE_DIR}/SKO_with_taxonomy.pkl')
gen_data_df = pd.read_pickle(f'{DATA_DIR}/generated_data_v3.pkl')
logger.info("Dataframes loaded successfully.")

fields = gen_data_df['Field'].unique()

logger.info("=" * 50)
logger.info(f"Starting fine-tuning for model: {args.model_id}...")
base_model = args.model_id
base_model_name = base_model.split('/')[1]
new_model_prefix = base_model_name + '_PL_'

# Load tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
logger.info("Tokenizer loaded successfully.")

# Define the directory to save the fine-tuned model checkpoints
MODEL_SAVE_DIR = f'{BASE_DIR}/models_MoE'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Check which fields are already trained for the model in MODEL_SAVE_DIR
trained_fields = [d for d in os.listdir(MODEL_SAVE_DIR) if d.startswith(new_model_prefix)]

# Iterate through each specified field and train the model separately
for field in fields:
    if f"{new_model_prefix}{field.replace(' ', '_')}" in trained_fields:
        logger.info(f"Skipping training for field: {field} as it is already trained.")
        continue

    logger.info(f"Starting training for field: {field}...")

    # Filter test data for the specific field
    test_data = base_df[base_df['Field'] == field]
    # Prepare the training data for the current field
    train_data = gen_data_df[gen_data_df['Field'] == field]

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_data[['title', 'abstract', 'Field', 'predicate_label']])
    test_dataset = Dataset.from_pandas(test_data[['title', 'abstract', 'Field', 'predicate_label']])
    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    dataset = dataset.map(format_chat_template)

    # Reload base model and apply LoRA configuration for each field to get independent training
    logger.info(f"Loading {base_model} for field: {field}...")
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
    logger.info(f"Base model and LoRA configuration loaded for field: {field}.")

    # Set eval_steps to half the length of the test data if it is greater than 5, else set to 5
    eval_steps = int(len(test_data) / 2) if len(test_data) > 5 else 5
    save_steps = int(len(test_data) / 2) if len(test_data) > 5 else 5
    training_arguments = TrainingArguments(
        output_dir=os.path.join(MODEL_SAVE_DIR, f'{new_model_prefix}{field.replace(" ", "_")}'),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        eval_strategy='steps',
        eval_steps=eval_steps,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="none",
        save_strategy="steps",
        save_steps=save_steps,
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

    # Train the model for the current field
    logger.info(f"Field {field} - Starting training...")
    trainer.train()
    logger.info(f"Field {field} - Training completed.")
    # Evaluate the model for the current field
    logger.info(f"Field {field} - Starting evaluation...")
    eval_results = trainer.evaluate()
    current_loss = eval_results['eval_loss']
    logger.info(f"Field {field} - Evaluation completed. Loss: {current_loss}\n")

    # Deleting the model and trainer
    del model
    del trainer
    # Run garbage collection
    gc.collect()
    # Clear the GPU cache
    torch.cuda.empty_cache()
