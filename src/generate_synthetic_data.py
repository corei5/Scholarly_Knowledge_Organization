import pandas as pd
import os
import re
import logging
from tqdm import tqdm
from openai import OpenAI
import json
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_generation_with_evaluation.log"),
    ]
)
logger = logging.getLogger(__name__)

# Helper to configure constants
def configure_paths(base_dir):
    data_dir = f'{base_dir}/data'
    output_file = f'{data_dir}/generated_data_with_refinement.pkl'
    model_call_count_file = f'{data_dir}/model_call_count.json'
    return data_dir, output_file, model_call_count_file

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created data directory at {directory}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Data generation and evaluation script.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for the project.")
    parser.add_argument('--min_count', type=int, default=100, help="Minimum count per field.")
    parser.add_argument('--iterations', type=int, default=3, help="Number of refinement iterations.")
    args = parser.parse_args()

    # Configure paths and constants
    BASE_DIR = args.base_dir
    MINIMUM_COUNT_PER_FIELD = args.min_count
    REFINEMENT_ITERATIONS = args.iterations
    DATA_DIR, OUTPUT_FILE, MODEL_CALL_COUNT_FILE = configure_paths(BASE_DIR)

    ensure_directory_exists(DATA_DIR)

    # Load OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
    client = OpenAI(api_key=api_key)
    logging.info("Loaded OpenAI API key from environment variables.")

    # Load or initialize model call count
    if os.path.exists(MODEL_CALL_COUNT_FILE):
        with open(MODEL_CALL_COUNT_FILE, 'r') as f:
            model_call_count = json.load(f)
            logging.info(f"Loaded model call count from {MODEL_CALL_COUNT_FILE}: {model_call_count}")
    else:
        model_call_count = {"gpt-4o-mini": 0, "gpt-4o": 0}
        logging.info("No model call count file found. Initialized model call count.")

    # Helper functions
    def get_response(prompt, model='gpt-4o-mini', retries=3):
        """
        Communicate with OpenAI API and get a response for a given prompt.
        """
        global model_call_count
        model_call_count[model] += 1

        for attempt in range(retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an Expert in different scientific fields."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"API error on attempt {attempt + 1} for model '{model}': {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        logger.error(f"All attempts failed for model '{model}'")
        return None

    def parse_initial_response(response):
        """
        Parse the initial response to extract title and abstract in structured format.
        """
        title_match = re.search(r"Title:\s*(.*)", response)
        abstract_match = re.search(r"Abstract:\s*(.*)", response)

        title = title_match.group(1).strip() if title_match else "Unknown Title"
        abstract = abstract_match.group(1).strip() if abstract_match else "Unknown Abstract"

        return title, abstract

    def evaluate_draft(title, abstract):
        """
        Evaluate a draft title and abstract and return an overall score.
        """
        prompt = f'''Evaluate the following research paper components for their relevance, clarity, and originality. 
        Provide a score from 0 to 10 for each criterion (relevance, clarity, originality) based on the provided guidelines, 
        and offer brief feedback on each component.

        Research Paper Components:
        Title: {title}
        Abstract: {abstract}

        Evaluation Guidelines:
        1. **Relevance**: Assess whether the title and abstract accurately represent the topic, key concepts, and purpose of the research.
        2. **Clarity**: Determine if the components are clear, concise, and well-structured, using language appropriate for an academic audience.
        3. **Originality**: Evaluate the uniqueness of the expression, considering if it adds distinctive value or perspective.

        Please provide the evaluation in the following format:

        Title:
        - Relevance Score: [0-10], Feedback: <Brief feedback on relevance>
        - Clarity Score: [0-10], Feedback: <Brief feedback on clarity>
        - Originality Score: [0-10], Feedback: <Brief feedback on originality>

        Abstract:
        - Relevance Score: [0-10], Feedback: <Brief feedback on relevance>
        - Clarity Score: [0-10], Feedback: <Brief feedback on clarity>
        - Originality Score: [0-10], Feedback: <Brief feedback on originality>

        Overall Score: [0-10]
        '''

        response = get_response(prompt, model="gpt-4o")
        if response:
            match = re.search(r"Overall Score:\s*([0-9]+(?:\.[0-9]*)?)", response)
            if match:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    logging.info(f"Draft evaluated with overall score: {score}")
                    return score
        logging.warning("Failed to parse overall score, using fallback value")
        return 5