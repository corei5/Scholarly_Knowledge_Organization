import os
import json
import pandas as pd
import joblib
import logging
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import gc
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import re

def extract_predicate_label(text):
    """
    Extracts list-like structures from the text, combines multiple lists if found,
    ensures unique values, and returns a clean list.

    Args:
        text (str): The input text containing lists.

    Returns:
        list: A cleaned, unique list of predicate labels.
    """
    if not text:
        return []

    # Regular expression to match list-like structures
    pattern = r'\[([^\[\]]+)\]'
    matches = re.findall(pattern, text)

    combined_labels = set()  # Use a set to ensure uniqueness

    for match in matches:
        # Split items in the matched list and clean them
        items = [item.strip().strip('"').strip("'") for item in match.split(',')]
        combined_labels.update(items)  # Add to the set for uniqueness

    # Return a sorted list of unique labels
    return sorted(combined_labels)


# Argument parser to pass model_id as a script parameter
parser = argparse.ArgumentParser(description='Evaluate different LLMs as expert models in an MoE framework.')
parser.add_argument('--model_id', type=str, required=True, help='Model ID to use as the expert model (e.g., meta-llama/Llama-3.1-8B-Instruct)')
parser.add_argument('--base_dir', type=str, required=True, help='Base directory for data and models.')
parser.add_argument('--version', type=str, required=True, help='Version of the MoE framework to use.')
args = parser.parse_args()
model_id = args.model_id
base_dir = args.base_dir
version = args.version
model_name = model_id.split('/')[1]

# Load data and models
DATA_DIR = f'{base_dir}/data'
SYM_MODELS_DIR = f'{base_dir}/models'
MOE_MODELS_DIR = os.path.join(base_dir, f'models_MoE_{version}')

logging.info("Loading data... from %s", DATA_DIR)

# Load the best gatekeeper model (SVM classifier)
gatekeeper_model_path = os.path.join(SYM_MODELS_DIR, 'best_svm_model.pkl')
logging.info("Loading gatekeeper model from %s", gatekeeper_model_path)
gatekeeper_model = joblib.load(gatekeeper_model_path)
logging.info("Gatekeeper model loaded successfully.")

# Load best checkpoints for field-specific expert models
best_checkpoints_path = os.path.join(base_dir, f'best_checkpoints_MoE_{version}.pkl')
logging.info("Loading best checkpoints from %s", best_checkpoints_path)
best_checkpoints_df = pd.read_pickle(best_checkpoints_path)
logging.info("Best checkpoints loaded successfully.")

# Define routing mechanism based on gatekeeper model
# Load the dataset to be classified and routed
base_df = pd.read_pickle(f'{base_dir}/SKO_with_taxonomy.pkl')
logging.info("Base dataset loaded successfully.")

# Test on a subset of the data
test_df = base_df[['title', 'abstract', 'Field', 'predicate_label']]#.head(10) # Test on the first 10 rows
logging.info("Selected the first 10 rows from the base dataset for testing.")

# Load the TF-IDF vectorizer used for training the gatekeeper model
tfidf_vectorizer_path = os.path.join(SYM_MODELS_DIR, 'tfidf_vectorizer.pkl')
logging.info("Loading TF-IDF vectorizer from %s", tfidf_vectorizer_path)
tfidf = joblib.load(tfidf_vectorizer_path)
logging.info("TF-IDF vectorizer loaded successfully.")

# Transform the test data using the loaded TF-IDF vectorizer
logging.info("Vectorizing text using the loaded TF-IDF vectorizer...")
X_test = tfidf.transform(test_df['title'] + ' ' + test_df['abstract'])
logging.info("Text vectorization using TF-IDF completed.")

# Apply gatekeeper model to predict field
test_df['Field_Prediction'] = gatekeeper_model.predict(X_test)
logging.info("Gatekeeper model predictions completed.")



# Load the base model once
logging.info("Loading base model: %s", model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
logging.info("Base model loaded successfully.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
logging.info("Tokenizer loaded successfully.")

# set output directory and file
# output_dir = os.path.join(BASE_DIR, 'MoE_output_v3')
output_dir = os.path.join(base_dir, f'MoE_output_{version}')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'MoE_{model_name}.pkl')

try:
    logging.info("Loading previously saved responses from %s", output_path)
    output_df = pd.read_pickle(output_path)
    # Filter out rows with None in 'Expert_Response'
    incomplete_rows = output_df[output_df['Expert_Response'].isna()]
except FileNotFoundError:
    logging.info("No previous output found. Starting from scratch.")
    output_df = test_df.copy()
    output_df['Expert_Response'] = None
    output_df['Response_Time'] = None
    output_df[f"MoE_predicate_label_{model_name}_extracted"] = None
    incomplete_rows = output_df

# Iterate through rows to classify and route tasks from where it left off
logging.info("Starting MoE mechanism - Routing tasks to field-specific models.")
for idx, row in incomplete_rows.iterrows():
    try:
        field = row['Field_Prediction']
        checkpoint_info = best_checkpoints_df[(best_checkpoints_df['Field'] == field) & (best_checkpoints_df['Model'] == model_name)]
        if not checkpoint_info.empty:
            logging.info("Found checkpoint information for field: %s", field)

            # Load the best checkpoint for the predicted field
            best_checkpoint = os.path.join(MOE_MODELS_DIR, checkpoint_info['Best_Checkpoint'].values[0])

            logging.info("Loading fine-tuned weights for model: %s for field: %s", model_name, field)

            # Track time taken to load fine-tuned weights and generate response
            start_time = time.time()
            fine_tuned_model = PeftModel.from_pretrained(model, best_checkpoint)
            logging.info("Fine-tuned model loaded successfully for model: %s", model_name)

            # Define prompt
            prompt = f"Based on the provided information, determine the most appropriate Predicate Labels. Finally, provide the response in JSON format.\nPaper Title: {row['title']}\nAbstract: {row['abstract']}\npredicate_labels:"

            # Create a text generation pipeline
            logging.info("Creating text generation pipeline for model: %s", model_name)
            pipe = pipeline(
                task="text-generation",
                model=fine_tuned_model,
                tokenizer=tokenizer,
                eos_token_id=fine_tuned_model.config.eos_token_id,
                max_new_tokens=512,
                return_full_text=False
            )

            # Generate response using the expert model
            response = pipe(prompt)[0]['generated_text']
            end_time = time.time()
            response_time = end_time - start_time
            logging.info("Time taken to load weights and generate response for index %d: %.2f seconds", idx, response_time)

            # Update DataFrame directly
            output_df.at[idx, 'Expert_Response'] = response
            output_df.at[idx, 'Response_Time'] = response_time
            output_df.at[idx, f"MoE_predicate_label_{model_name}_extracted"] = extract_predicate_label(response)

            del fine_tuned_model
            torch.cuda.empty_cache()
            gc.collect()

        else:
            logging.warning("No checkpoint found for field: %s. Appending None as response.", field)
            output_df.loc[idx, 'Expert_Response'] = None
            output_df.loc[idx, 'Response_Time'] = None
            output_df.loc[idx, f"MoE_predicate_label_{model_name}_extracted"] = None

    except Exception as e:
        logging.error("Error occurred at index %d: %s", idx, str(e))
        # Save the progress made until the error
        output_df.to_pickle(output_path)
        logging.info("Progress saved to %s after encountering an error.", output_path)
        # Optionally, re-raise the error or continue
        continue

    # Save the output after each response generation
    output_df.to_pickle(output_path)
    logging.info("Intermediate MoE output saved to %s", output_path)

# Save final results
output_df.to_pickle(output_path)
logging.info("MoE output saved to %s", output_path)

print("MoE response generated and results saved.")
