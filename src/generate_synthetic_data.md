# Data Generation and Evaluation Script

This script generates synthetic data for research purposes, evaluates it based on specific criteria, and refines the generated data iteratively.

## Prerequisites

1. Python 3.8 or higher
2. Install required packages: `pandas`, `tqdm`, `openai`, `json`, `argparse`.

## Environment Variables

Ensure the `OPENAI_API_KEY` is set in your environment:
```bash
export OPENAI_API_KEY=<your-api-key>
```

## Usage

Run the script with the following arguments:

```bash
python script.py --base_dir <BASE_DIR> --min_count <MIN_COUNT> --iterations <ITERATIONS>
```

- `--base_dir`: The base directory where data is stored and generated.
- `--min_count`: Minimum count of entries per field (default: 100).
- `--iterations`: Number of refinement iterations (default: 3).

## Example

```bash
python script.py --base_dir /path/to/project --min_count 150 --iterations 5
```

This will ensure each field in the dataset has at least 150 entries with up to 5 iterations for refinement.
