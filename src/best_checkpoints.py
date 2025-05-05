import os
import json
import pandas as pd
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_paths(base_dir, version):
    data_dir = f"{base_dir}/data"
    all_model_dir = f"{base_dir}/models_MoE_{version}"
    best_model_dir = f"{base_dir}/models_MoE_{version}_best"
    return data_dir, all_model_dir, best_model_dir

def load_data(base_dir):
    """Load generated data from a pickle file."""
    logging.info("Loading data...")
    try:
        return pd.read_pickle(f'{base_dir}/SKO_with_taxonomy.pkl')
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def get_model_and_field_combinations(gen_data_df, model_ids):
    """Generate combinations of model IDs and fields."""
    gen_fields = gen_data_df['Field'].unique()
    return [(model_id, field) for model_id in model_ids for field in gen_fields]

def find_best_checkpoint(model_name, field, all_model_dir):
    """Find the best checkpoint for a given model and field."""
    field_path = os.path.join(all_model_dir, f"{model_name}_PL_{field.replace(' ', '_')}")
    if not os.path.exists(field_path):
        logging.info(f"Directory does not exist for Model: {model_name}, Field: {field}")
        return None

    best_metric = float('inf')
    best_checkpoint = None

    for checkpoint_dir in os.listdir(field_path):
        checkpoint_path = os.path.join(field_path, checkpoint_dir)
        trainer_state_file = os.path.join(checkpoint_path, 'trainer_state.json')

        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                metric = trainer_state.get("best_metric", float('inf'))
                if metric < best_metric:
                    best_metric = metric
                    best_checkpoint = trainer_state.get("best_model_checkpoint", checkpoint_path)

    return best_checkpoint

def process_models_and_fields(model_ids, gen_data_df, all_model_dir):
    """Process models and fields to find best checkpoints."""
    best_checkpoints = []
    missing_fields = []
    combinations = get_model_and_field_combinations(gen_data_df, model_ids)

    for model_id, field in combinations:
        model_name = model_id.split('/')[-1]
        best_checkpoint = find_best_checkpoint(model_name, field, all_model_dir)

        if best_checkpoint:
            best_checkpoints.append({
                'Model': model_name,
                'Field': field,
                'Best_Checkpoint': best_checkpoint
            })
        else:
            missing_fields.append({'Model': model_name, 'Field': field})

    return best_checkpoints, missing_fields

def save_results(best_checkpoints, missing_fields, base_dir, version):
    """Save results to pickle files."""
    try:
        best_checkpoints_df = pd.DataFrame(best_checkpoints)
        best_checkpoints_df.to_pickle(os.path.join(base_dir, f'best_checkpoints_MoE_{version}.pkl'))
        logging.info(f"Best checkpoints saved to best_checkpoints_MoE_{version}.pkl")

        missing_fields_df = pd.DataFrame(missing_fields)
        missing_fields_df.to_pickle(os.path.join(base_dir, 'missing_fields_MoE.pkl'))
        logging.info("Missing fields saved to missing_fields_MoE.pkl")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process models and fields to find best checkpoints.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for the project.")
    parser.add_argument('--version', type=str, required=True, help="Version of the MoE models.")
    args = parser.parse_args()

    # Configure paths
    DATA_DIR, ALL_MODEL_DIR, BEST_MODEL_DIR = configure_paths(args.base_dir, args.version)

    # Load data
    gen_data_df = load_data(args.base_dir)

    # Define model IDs
    model_ids = [
        'meta-llama/Llama-3.1-8B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'HuggingFaceH4/zephyr-7b-beta',
        'microsoft/Phi-3.5-mini-instruct',
        'google/gemma-2-9b-it',
    ]

    # Process models and fields
    best_checkpoints, missing_fields = process_models_and_fields(model_ids, gen_data_df, ALL_MODEL_DIR)

    # Save results
    save_results(best_checkpoints, missing_fields, args.base_dir, args.version)
