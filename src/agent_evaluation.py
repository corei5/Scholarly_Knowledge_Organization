import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()

# Load API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

# Create the OpenAI object
client = OpenAI(api_key=api_key)

# Function to create predicate label evaluation prompt
def create_predicate_label_evaluation_prompt(title, abstract, expected_labels, response):
    return f"""
    Your task is to evaluate the correctness of the predicate labels identified by a language model based on the provided title and abstract of a research paper.

    Here is the paper’s title: {title}
    Here is the paper's abstract: {abstract}

    Expected Predicate Label(s): {expected_labels}

    Language model's identified predicate labels: {response}

    Instructions:
    1. Compare the language model's identified predicate labels to the expected predicate labels.
    2. For each label identified by the language model:
       - Explain why it is correct or incorrect based on the title and abstract.
       - Match it to the expected predicate labels if it aligns with the context of the title and abstract.
    3. Return the list of correctly predicted labels **from the expected predicate labels** in the following format:

    Correctly Predicted Labels: ["label1", "label2", ...]
    """

# Function to create research field evaluation prompt
def create_research_field_evaluation_prompt(title, abstract, expected_label, response):
    return f"""
    Your task is to evaluate the correctness of the research field label identified by a language model based on the provided title and abstract of a research paper.

    Here is the paper’s title: {title}
    Here is the paper's abstract: {abstract}

    Expected Research Field Label: {expected_label}

    Language model's predicted research field label: {response}

    Instructions:
    1. Critically evaluate the predicted research field label by comparing it to the expected label.
    2. Assess whether the predicted research field label is directly supported by explicit information in the title or abstract.
    3. Penalize labels that:
       - Are overly broad or generic (e.g., "Science" for "Quantum Computing in Medicine").
       - Contradict the primary focus of the title or abstract.
       - Omit key details necessary for accurately identifying the research field.
    4. Provide reasoning for why the predicted research field label is correct or incorrect. Include specific evidence or examples from the abstract and title to support your decision.
    5. Finally, provide your decision in the following strict format:

    Decision: "Correct" or "Incorrect"
    """

# Function to get the evaluation from GPT-4o API
def evaluate_response(prompt_content):
    logging.info(f"Prompt: {prompt_content[:500]}...")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an Expert Evaluator."},
                {"role": "user", "content": prompt_content}
            ]
        )
        response = completion.choices[0].message.content.strip()
        logging.info(f"Response: {response[:100]}...")
        return response
    except Exception as e:
        logging.error(f"API error: {e}")
        return None

# Function to evaluate a single row
def evaluate_row(row, prompt_type, prompt_fn, model_name):
    try:
        if 'predicate_label' in prompt_type:
            label_col = 'predicate_label'
        elif 'research_field' in prompt_type:
            label_col = 'Field'
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        model_output = row.get(f'{prompt_type}_{model_name}')
        if pd.isna(model_output):
            return None
        # Create the prompt
        prompt = prompt_fn(row['title'], row['abstract'], row[label_col], model_output)
        return evaluate_response(prompt)
    except Exception as e:
        logging.error(f"Error evaluating row: {e}")
        return None

# Function to evaluate model responses with parallelism
def evaluate_model_responses(df, model_name, file_path, prompt_types_map):
    for prompt_type, prompt_fn in prompt_types_map:
        eval_col = f'{prompt_type}_gpt-4o_evaluation'

        if eval_col not in df.columns:
            df[eval_col] = None
            logging.info(f"Added column {eval_col} to DataFrame.")

        rows_to_evaluate = [
            (row, prompt_type, prompt_fn, model_name)
            for idx, row in df.iterrows()
            if pd.notna(row.get(f'{prompt_type}_{model_name}')) and pd.isna(row.get(eval_col))
        ]

        max_threads = min(8, multiprocessing.cpu_count())
        with ThreadPoolExecutor(max_threads) as executor:
            results = list(tqdm(
                executor.map(lambda args: evaluate_row(*args), rows_to_evaluate),
                total=len(rows_to_evaluate),
                desc=f"Evaluating {prompt_type} for {model_name}"
            ))

        for (row, _, _, _), result in zip(rows_to_evaluate, results):
            idx = row.name
            df.at[idx, eval_col] = result

        df.to_pickle(file_path)
        logging.info(f"Progress saved after evaluating {prompt_type} for {model_name}.")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Run model prompts and save responses.")
    parser.add_argument('--model_type', type=str, required=True, help="Type of model (MoE, FT, Base)")
    parser.add_argument('--model_id', type=str, required=True, help="Hugging Face model ID")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for input/output files")
    parser.add_argument('--version', type=str, default="v1", help="Version identifier")
    args = parser.parse_args()

    model_type = args.model_type
    model_id = args.model_id
    base_dir = args.base_dir
    version = args.version

    model_name = model_id.split('/')[-1]
    logging.info(f"Evaluating {model_type} model: {model_name}")

    if model_type == "MoE":
        prompt_types_map = [
            ('MoE_predicate_label', create_predicate_label_evaluation_prompt),
            ('MoE_research_field', create_research_field_evaluation_prompt),
        ]
    elif model_type == "FT":
        prompt_types_map = [
            ('fine_tuned_predicate_label', create_predicate_label_evaluation_prompt),
            ('fine_tuned_research_field', create_research_field_evaluation_prompt),
        ]
    elif model_type == "Base":
        prompt_types_map = [
            ('zero_shot_predicate_label', create_predicate_label_evaluation_prompt),
            ('zero_shot_research_field', create_research_field_evaluation_prompt),
            ('few_shot_predicate_label', create_predicate_label_evaluation_prompt),
            ('few_shot_research_field', create_research_field_evaluation_prompt),
            ('cot_predicate_label', create_predicate_label_evaluation_prompt),
            ('cot_research_field', create_research_field_evaluation_prompt),
            ('zero_shot_cot_predicate_label', create_predicate_label_evaluation_prompt),
            ('zero_shot_cot_research_field', create_research_field_evaluation_prompt),
            ('sci_predicate_label', create_predicate_label_evaluation_prompt),
            ('sci_research_field', create_research_field_evaluation_prompt),
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    output_dir = f"{base_dir}/output/{model_type}_{version}"
    evaluated_file_path = f"{output_dir}/{model_name}_gpt-4o_evaluated.pkl"
    initial_file_path = f"{output_dir}/{model_name}.pkl"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        if os.path.exists(evaluated_file_path):
            logging.info(f"Loading previous evaluation file: {evaluated_file_path}")
            df = pd.read_pickle(evaluated_file_path)
        else:
            logging.info(f"No previous evaluation found, loading initial file: {initial_file_path}")
            df = pd.read_pickle(initial_file_path)
            df = df.reset_index(drop=True)

    except FileNotFoundError:
        logging.error(f"File not found: {initial_file_path}. Ensure path is correct.")
        return

    evaluate_model_responses(df, model_name, evaluated_file_path, prompt_types_map)
    df.to_pickle(evaluated_file_path)
    logging.info(f"Final evaluations saved to {evaluated_file_path}")

if __name__ == "__main__":
    main()
    logging.info(f"Time taken: {time.time() - start_time} seconds")
