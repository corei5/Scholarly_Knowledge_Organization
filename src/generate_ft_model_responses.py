import os
import time
import pickle
import logging
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHECKPOINT_FILE = "best_checkpoints.pkl"
DEFAULT_TEST_DATA_FILE = "SKO_with_taxonomy.pkl"

# Function to load model checkpoint
def load_model_checkpoint(base_dir, model_name):
    checkpoint_path = os.path.join(base_dir, f"{model_name}", DEFAULT_CHECKPOINT_FILE)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    return checkpoint_data

# Function to load tokenizer and model
def load_model_and_tokenizer(checkpoint_data):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_data["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_data["model_id"], trust_remote_code=True, device_map="auto")
    return model, tokenizer

# Function to create task-specific prompt
def create_prompt(task, title, abstract):
    if task == "rf":
        return f"""
        Based on the following title and abstract, identify the most appropriate Research Field:

        Title: {title}
        Abstract: {abstract}

        Research Field:
        """
    elif task == "pl":
        return f"""
        Based on the following title and abstract, generate the most appropriate Predicate Labels in JSON format:

        Title: {title}
        Abstract: {abstract}

        Predicate Labels:
        """
    else:
        raise ValueError(f"Unsupported task type: {task}")

# Function to generate responses
def generate_responses(task, model, tokenizer, test_data, output_file, max_tokens=512):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    responses = []
    times = []

    logger.info("Generating responses...")
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        title = row["title"]
        abstract = row["abstract"]
        prompt = create_prompt(task, title, abstract)

        start_time = time.time()
        try:
            response = pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)[0]["generated_text"]
        except Exception as e:
            logger.error(f"Error generating response for row {idx}: {e}")
            response = None

        elapsed_time = time.time() - start_time
        responses.append((row["title"], response))
        times.append(elapsed_time)

    logger.info("Saving responses...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump({"responses": responses, "times": times}, f)
    logger.info(f"Responses saved to {output_file}")

# Main function
def main(base_dir, task, model_name, output_dir, max_tokens=512):
    # Load model checkpoint
    checkpoint_data = load_model_checkpoint(base_dir, model_name)

    # Load tokenizer and model
    model, tokenizer = load_model_and_tokenizer(checkpoint_data)

    # Load test data
    test_data_file = os.path.join(base_dir, DEFAULT_TEST_DATA_FILE)
    logger.info(f"Loading test data from {test_data_file}")
    test_data = pd.read_pickle(test_data_file)

    # Generate and save responses
    output_file = os.path.join(output_dir, f"{model_name}_{task}_responses.pkl")
    generate_responses(task, model, tokenizer, test_data, output_file, max_tokens=max_tokens)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate model responses for RF and PL tasks.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing model and data files.")
    parser.add_argument("--task", type=str, required=True, choices=["rf", "pl"], help="Task type: rf (Research Field) or pl (Predicate Label).")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the fine-tuned model to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output responses.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens for the generated response.")

    args = parser.parse_args()
    main(args.base_dir, args.task, args.model_name, args.output_dir, args.max_tokens)
