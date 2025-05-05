import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------- Utility Functions -------------------

def load_data(train_file, test_file):
    """
    Load training and testing datasets from the provided paths.

    Args:
        train_file (str): File path for the training dataset.
        test_file (str): File path for the testing dataset.

    Returns:
        tuple: Loaded training and testing datasets as Pandas DataFrames.
    """
    logging.info("Loading datasets...")
    train_df = pd.read_pickle(train_file)
    test_df = pd.read_pickle(test_file)
    logging.info("Datasets loaded successfully.")
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Preprocess data by combining title and abstract columns.

    Args:
        train_df (DataFrame): The training dataset.
        test_df (DataFrame): The testing dataset.

    Returns:
        tuple: Preprocessed training and testing data.
    """
    logging.info("Preprocessing datasets...")
    train_df['text'] = train_df['title'] + ' ' + train_df['abstract']
    test_df['text'] = test_df['title'] + ' ' + test_df['abstract']

    X_train = train_df['text']
    y_train = train_df['Field']
    X_test = test_df['text']
    y_test = test_df['Field']

    return X_train, y_train, X_test, y_test

def vectorize_text(X_train, X_test, models_dir, max_features=5000):
    """
    Vectorize text data using TF-IDF.

    Args:
        X_train (Series): Training text data.
        X_test (Series): Testing text data.
        models_dir (str): Directory to save the TF-IDF vectorizer.
        max_features (int): Maximum number of features for TF-IDF.

    Returns:
        tuple: Transformed training and testing data.
    """
    logging.info("Vectorizing text data using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(tfidf, vectorizer_path)
    logging.info("TF-IDF vectorizer saved at %s", vectorizer_path)

    return X_train_tfidf, X_test_tfidf

def train_and_evaluate_classifiers(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """
    Train and evaluate multiple SVM classifiers.

    Args:
        X_train_tfidf (sparse matrix): TF-IDF transformed training data.
        y_train (Series): Training labels.
        X_test_tfidf (sparse matrix): TF-IDF transformed testing data.
        y_test (Series): Testing labels.

    Returns:
        tuple: List of accuracy results, best classifier, and its accuracy.
    """
    svm_classifiers = {
        'SVM (C=1.0, Kernel=rbf)': SVC(C=1.0, kernel='rbf', random_state=42),
        'SVM (C=0.5, Kernel=linear)': SVC(C=0.5, kernel='linear', random_state=42),
        'SVM (C=1.5, Kernel=poly, Degree=3)': SVC(C=1.5, kernel='poly', degree=3, random_state=42),
        'SVM (C=1.0, Kernel=sigmoid)': SVC(C=1.0, kernel='sigmoid', random_state=42)
    }

    accuracy_results = []
    best_model = None
    best_accuracy = 0

    for name, clf in svm_classifiers.items():
        logging.info(f"Training classifier: {name}")
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy for {name}: {accuracy}")
        logging.info(f"Classification Report for {name}:
{classification_report(y_test, y_pred)}")

        accuracy_results.append({'Classifier': name, 'Accuracy': accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf

    return accuracy_results, best_model, best_accuracy

def save_results(accuracy_results, best_model, scores_dir, models_dir):
    """
    Save accuracy results and the best model.

    Args:
        accuracy_results (list): List of accuracy results for each classifier.
        best_model (object): The best-performing classifier.
        scores_dir (str): Directory to save accuracy results.
        models_dir (str): Directory to save the best model.
    """
    logging.info("Saving results...")

    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    accuracy_df = pd.DataFrame(accuracy_results)
    accuracy_path = os.path.join(scores_dir, 'svm_classification_scores.pkl')
    accuracy_df.to_pickle(accuracy_path)
    logging.info("Accuracy results saved at %s", accuracy_path)

    if best_model is not None:
        model_path = os.path.join(models_dir, 'best_svm_model.pkl')
        joblib.dump(best_model, model_path)
        logging.info("Best model saved at %s", model_path)

# ------------------- Main Execution -------------------
def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SVM classifiers on text data.')
    parser.add_argument('--train_file', type=str, required=True, help='File path for the training dataset.')
    parser.add_argument('--test_file', type=str, required=True, help='File path for the testing dataset.')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory to save models and vectorizer.')
    parser.add_argument('--scores_dir', type=str, required=True, help='Directory to save classification scores.')
    args = parser.parse_args()

    TRAIN_FILE = args.train_file
    TEST_FILE = args.test_file
    MODELS_DIR = args.models_dir
    SCORES_DIR = args.scores_dir

    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

    X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test, MODELS_DIR)

    accuracy_results, best_model, best_accuracy = train_and_evaluate_classifiers(X_train_tfidf, y_train, X_test_tfidf, y_test)

    save_results(accuracy_results, best_model, SCORES_DIR, MODELS_DIR)

if __name__ == "__main__":
    main()
