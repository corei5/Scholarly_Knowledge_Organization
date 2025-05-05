import pandas as pd
import numpy as np
import transformers
import torch
from tqdm import tqdm
import time
import os
import argparse
import logging
import re
from collections import Counter
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- Model Initialization ----------------------
def initialize_model_pipeline(model_id):
    try:
        logging.info(f"Initializing model pipeline for {model_id}")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            return_full_text=False
        )
        logging.info(f"Model pipeline for {model_id} initialized successfully")
        return pipeline
    except Exception as e:
        logging.error(f"Error initializing model {model_id}: {e}")
        return None

# ---------------------- Response Generation ----------------------
def get_model_response(pipeline, prompt, max_new_tokens=1024):
    if pipeline is None:
        logging.error("Pipeline not initialized. Cannot generate response.")
        return None

    try:
        logging.info("Generating response from the model.")
        response = pipeline(prompt, max_new_tokens=max_new_tokens)
        return response[0]["generated_text"]
    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        return None

# ---------------------- Extraction Functions ----------------------
def extract_predicate_label(response, expected_title):
    title_pattern = r'"paper_title":\s*"(.*?)"'
    labels_pattern = r'"predicate_label(?:s)?":\s*\[([\s\S]*?)\]'

    titles = re.findall(title_pattern, response, re.DOTALL)
    labels_matches = re.findall(labels_pattern, response)

    titles, labels_matches = sync_extraction_lengths(titles, labels_matches)

    matched_responses = find_matching_labels(titles, labels_matches, expected_title)

    if matched_responses:
        return list(Counter(matched_responses).most_common(1)[0][0])

    return []

def extract_research_field(response, expected_title):
    pattern = r'"paper_title":\s*"(.*?)".*?"research_field":\s*"(.*?)"'
    matches = re.findall(pattern, response, re.DOTALL)

    for title, research_field in matches:
        if is_close_match(title, expected_title):
            return research_field

    return None

# ---------------------- Helper Functions ----------------------
def sync_extraction_lengths(titles, labels_matches):
    if len(titles) != len(labels_matches):
        min_length = min(len(titles), len(labels_matches))
        titles = titles[:min_length]
        labels_matches = labels_matches[:min_length]
    return titles, labels_matches

def find_matching_labels(titles, labels_matches, expected_title):
    def is_close_match(title1, title2, threshold=0.5):
        return SequenceMatcher(None, title1.lower().replace('\n', ' '), title2.lower().replace('\n', ' ')).ratio() >= threshold

    matched_responses = []
    for i, title in enumerate(titles):
        if is_close_match(title, expected_title):
            predicate_labels = clean_labels(labels_matches[i])
            matched_responses.append(tuple(predicate_labels))
    return matched_responses

def clean_labels(labels):
    return [label.strip() for label in re.sub(r'[\n\s]+', '', labels).replace('"', '').split(',')]

def is_close_match(title1, title2, threshold=0.5):
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio() >= threshold

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------- DataFrame Processing ----------------------
def process_dataframe(df, pipeline, model_name, filename):
    prompt_types = [
        ("sci_predicate_label", extract_predicate_label),
        ("sci_research_field", extract_research_field)
    ]

    for prompt_type, extract_func in prompt_types:
        logging.info(f"Processing {prompt_type} prompts.")
        df = process_prompts(df, pipeline, model_name, filename, prompt_type, extract_func)

    logging.info(f"All responses for {model_name} saved to {filename}")
    return df

def process_prompts(df, pipeline, model_name, filename, prompt_type, extract_func):
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {prompt_type} prompts'):
        if pd.isna(df.at[idx, f'{prompt_type}_{model_name}_extracted']):
            process_single_row(df, idx, row, pipeline, model_name, filename, prompt_type, extract_func)
        else:
            logging.info(f"Skipping index {idx} for {prompt_type} as it is already processed.")
    return df

def process_single_row(df, idx, row, pipeline, model_name, filename, prompt_type, extract_func):
    title = row['paper_title']
    abstract = row['abstract']

    for attempt in range(3):
        prompt = row[f"{prompt_type}_prompt"]
        start_time = time.time()
        response = get_model_response(pipeline, prompt)
        response_time = time.time() - start_time

        df.at[idx, f'{prompt_type}_{model_name}'] = response
        df.at[idx, f'{prompt_type}_{model_name}_time'] = response_time

        extracted_data = extract_func(response, title)
        if extracted_data:
            df.at[idx, f'{prompt_type}_{model_name}_extracted'] = extracted_data
            logging.info(f"Extraction successful for index {idx}.")
            break
        else:
            logging.warning(f"Extraction failed on attempt {attempt + 1} for index {idx}.")
            set_seed(attempt)

    if pd.isna(df.at[idx, f'{prompt_type}_{model_name}_extracted']):
        logging.error(f"Extraction failed after 3 attempts for index {idx}.")

    save_progress(df, filename, idx, prompt_type)

def save_progress(df, filename, idx, prompt_type):
    df.to_pickle(filename)
    logging.info(f"Progress saved after processing index {idx} for {prompt_type}.")

# ---------------------- Main Function ----------------------
def main():
    parser = argparse.ArgumentParser(description="Run model prompts and save responses.")
    parser.add_argument('--model_id', type=str, required=True, help="Hugging Face model ID")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for input and output files")
    parser.add_argument('--output_subdir', type=str, required=True, help="Subdirectory for saving output files")
    parser.add_argument('--default_file', type=str, required=True, help="Path to the default input DataFrame file")
    args = parser.parse_args()

    model_id = args.model_id
    base_dir = args.base_dir
    save_directory = os.path.join(base_dir, args.output_subdir)
    filename = os.path.join(save_directory, f'{model_id.split('/')[-1]}.pkl')

    os.makedirs(save_directory, exist_ok=True)

    pipeline = initialize_model_pipeline(model_id)

    df = load_dataframe(filename, args.default_file)
    df_with_responses = process_dataframe(df, pipeline, model_id.split('/')[-1], filename)
    logging.info("Script execution completed successfully.")

def load_dataframe(filename, default_file):
    try:
        logging.info(f"Trying to load dataframe from {filename}")
        df = pd.read_pickle(filename)
    except Exception as e:
        logging.error(f"Error loading dataframe: {e}")
        logging.info("Loading from default file")
        df = pd.read_pickle(default_file)
    return df

if __name__ == "__main__":
    main()
