import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import os
import argparse

def calculate_accuracy_pl(df, true_col, pred_col, threshold=0.90):
    accuracies = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        true_labels = row[true_col]
        predicted_labels = row[pred_col]
        if not true_labels or not predicted_labels:
            accuracies.append(0)
            continue
        true_embeddings = model.encode(true_labels)
        predicted_embeddings = model.encode(predicted_labels)
        matched = 0
        for pred_emb in predicted_embeddings:
            for true_emb in true_embeddings:
                similarity = pred_emb @ true_emb.T
                if similarity >= threshold:
                    matched += 1
                    break
        row_accuracy = matched / max(len(true_labels), len(predicted_labels))
        accuracies.append(row_accuracy)
    df[f"{pred_col}_accuracy"] = accuracies
    return df

def calculate_metrics_pl(df, true_col, pred_col, threshold=0.90):
    true_positive, false_positive, false_negative = 0, 0, 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        true_labels = row[true_col]
        predicted_labels = row[pred_col]
        if not predicted_labels:
            false_negative += len(true_labels)
            continue
        true_embeddings = model.encode(true_labels)
        predicted_embeddings = model.encode(predicted_labels)
        matched_true = set()
        for pred_emb in predicted_embeddings:
            matched = False
            for i, true_emb in enumerate(true_embeddings):
                similarity = pred_emb @ true_emb.T
                if similarity >= threshold and i not in matched_true:
                    true_positive += 1
                    matched_true.add(i)
                    matched = True
                    break
            if not matched:
                false_positive += 1
        false_negative += len(true_labels) - len(matched_true)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def calculate_accuracy_rf(df, true_col, pred_col, threshold=0.90):
    df[f"{pred_col}_is_correct"] = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        true_label = row[true_col]
        predicted_label = row[pred_col]
        if pd.isnull(predicted_label):
            continue
        true_embedding = model.encode([true_label])[0]
        predicted_embedding = model.encode([predicted_label])[0]
        similarity = true_embedding @ predicted_embedding.T
        if similarity >= threshold:
            df.loc[idx, f"{pred_col}_is_correct"] = 1
    return df

def evaluate_models(model_type, model_ids, prompt_types, base_dir, threshold=0.90, version="v1"):
    global model
    model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)
    model_output_dir = f"{base_dir}/{model_type}_output_{version}"
    scores_dir = f"{base_dir}/scores/{model_type}_{version}"
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    predicat_label_scores, research_field_scores = [], []
    start_time = time.time()

    for model_id in model_ids:
        predicat_label_score, research_field_score = {}, {}
        model_name = model_id.split('/')[-1]
        print(f"***** Evaluating {model_name} *****")
        predicat_label_score['model'] = model_name
        research_field_score['model'] = model_name

        df = pd.read_pickle(f"{model_output_dir}/{model_name}.pkl")
        for prompt_type in prompt_types:
            if 'predicate_label' in prompt_type:
                print(f"Processing {prompt_type}")
                predicat_label_score[f"{prompt_type}_error"] = df[f'{prompt_type}_{model_name}_extracted'].isnull().sum()
                df[f'{prompt_type}_{model_name}_precision'], df[f'{prompt_type}_{model_name}_recall'], df[f'{prompt_type}_{model_name}_f1'] = calculate_metrics_pl(df, 'predicate_label', f'{prompt_type}_{model_name}_extracted', threshold=threshold)
                predicat_label_score[f"{prompt_type}_avg_precision"] = df[f'{prompt_type}_{model_name}_precision'].mean()
                predicat_label_score[f"{prompt_type}_avg_recall"] = df[f'{prompt_type}_{model_name}_recall'].mean()
                predicat_label_score[f"{prompt_type}_avg_f1"] = df[f'{prompt_type}_{model_name}_f1'].mean()

            if 'research_field' in prompt_type:
                print(f"Processing {prompt_type}")
                research_field_score[f"{prompt_type}_error"] = df[f'{prompt_type}_{model_name}_extracted'].isnull().sum()
                df = calculate_accuracy_rf(df, 'Field', f'{prompt_type}_{model_name}_extracted', threshold=threshold)
                research_field_score[f"{prompt_type}_avg_accuracy"] = df[f"{prompt_type}_{model_name}_extracted_is_correct"].mean()

        predicat_label_scores.append(predicat_label_score)
        research_field_scores.append(research_field_score)

    pd.DataFrame(predicat_label_scores).to_pickle(f"{scores_dir}/predicate_label_scores.pkl")
    pd.DataFrame(research_field_scores).to_pickle(f"{scores_dir}/research_field_scores.pkl")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Models for Predicate Label and Research Field Tasks")
    parser.add_argument('--model_type', type=str, required=True, help="Type of model to evaluate (MoE, FT, Base)")
    parser.add_argument('--model_ids', nargs='+', required=True, help="List of model IDs to evaluate")
    parser.add_argument('--prompt_types', nargs='+', required=True, help="List of prompt types to evaluate")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for model outputs and scores")
    parser.add_argument('--threshold', type=float, default=0.90, help="Threshold for cosine similarity (default: 0.90)")
    parser.add_argument('--version', type=str, default="v1", help="Version identifier for the evaluation (default: v1)")

    args = parser.parse_args()

    evaluate_models(
        model_type=args.model_type,
        model_ids=args.model_ids,
        prompt_types=args.prompt_types,
        base_dir=args.base_dir,
        threshold=args.threshold,
        version=args.version
    )
