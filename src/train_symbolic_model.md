## Overview
This script trains and evaluates multiple classifiers on text data using TF-IDF features. The best-performing model and results are saved for future use.

## Usage
Run the script with the following arguments:
```bash
python script.py \
    --train_file <path_to_training_file> \
    --test_file <path_to_testing_file> \
    --models_dir <path_to_save_models> \
    --scores_dir <path_to_save_scores>
```

### Arguments
- `--train_file`: Path to the training dataset (pickle format).
- `--test_file`: Path to the testing dataset (pickle format).
- `--models_dir`: Directory to save the TF-IDF vectorizer and the best model.
- `--scores_dir`: Directory to save classification scores.

## Outputs
- **Best Model**: Saved as `best_svm_model.pkl` in the specified `models_dir`.
- **TF-IDF Vectorizer**: Saved as `tfidf_vectorizer.pkl` in the `models_dir`.
- **Classification Scores**: Saved as `svm_classification_scores.pkl` in the `scores_dir`.

