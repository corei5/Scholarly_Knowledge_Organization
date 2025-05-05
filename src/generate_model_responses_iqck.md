
## Overview


## Prerequisites
- Python 3.8 or later
- Required Python libraries:
  - pandas
  - numpy
  - transformers
  - torch
  - tqdm
  - argparse

Install dependencies using:
```bash
pip install pandas numpy transformers torch tqdm
```

## Usage
### Command-Line Arguments
The script uses the following arguments:

- `--model_id`: The Hugging Face model ID to use (e.g., `EleutherAI/gpt-neo-2.7B`).
- `--base_dir`: The base directory for input and output files.
- `--output_subdir`: The subdirectory within `base_dir` to save output files.
- `--default_file`: The path to the default input DataFrame file.

### Example Command
```bash
python generate_model_responses_iqck.py \
    --model_id 'meta-llama/Llama-3.1-8B-Instruct' \
    --base_dir /path/to/base_dir \
    --output_subdir output_sci/test_15 \
    --default_file /path/to/base_dir/SKO_with_IQCK_prompts.pkl
```

### Output
The processed DataFrame is saved as a pickle file in the specified `output_subdir` directory, named after the model ID (e.g., `gpt-neo-2.7B.pkl`).

## Key Functions
### 1. **initialize_model_pipeline**
Initializes the Hugging Face text-generation model pipeline.

### 2. **get_model_response**
Generates a response from the model for a given prompt.

### 3. **extract_predicate_label** / **extract_research_field**
Extracts specific fields from the model's response using regex and similarity matching.

### 4. **process_dataframe**
Processes the DataFrame, generates responses, and extracts relevant fields.

### 5. **save_progress**
Saves intermediate progress to avoid data loss during processing.

## File Structure
- **Script**: The Python script (`generate_model_responses_iqck.py`).
- **Data**: Input DataFrame (`default_file`) in pickle format.
- **Output**: Processed DataFrame saved as a pickle file in the specified output directory.

## Customization
To add new fields or customize extraction logic, update the `prompt_types` in `process_dataframe` and implement corresponding extraction functions.

## License
This script is provided as-is under an open-source license. Feel free to modify and use it in your projects.

---
For questions or support, feel free to contact the script maintainer.

