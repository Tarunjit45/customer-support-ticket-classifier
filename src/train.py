"""
Model Training Script for Customer Support Ticket Classification.

This script:
1. Loads historical customer support tickets from a CSV dataset.
2. Validates data integrity and cleans ticket text.
3. Splits data into stratified training and testing sets.
4. Builds a scikit-learn pipeline (TF-IDF Vectorizer + Logistic Regression).
5. Trains the classifier.
6. Evaluates the model on the test set.
7. Saves the trained model pipeline to disk using joblib.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ticket_classifier.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EVAL_PATH = os.path.join(OUTPUT_DIR, "evaluation.txt")


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Load dataset and perform validation checks."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = ["text", "category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset.")

    # Drop rows with missing text or category
    initial_count = len(df)
    df = df.dropna(subset=required_columns).copy()

    # Clean text: convert to string, strip whitespace
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # Drop empty strings
    df = df[(df["text"] != "") & (df["category"] != "")].copy()

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"]).copy()

    cleaned_count = len(df)
    if cleaned_count == 0:
        raise ValueError("Dataset is empty after cleaning.")

    print(f"Data loaded successfully: {cleaned_count} records (dropped {initial_count - cleaned_count} invalid/duplicate rows).")
    print(f"Categories ({df['category'].nunique()} total): {', '.join(sorted(df['category'].unique()))}")
    return df


def build_pipeline() -> Pipeline:
    """Build a TF-IDF + Logistic Regression classification pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        ))
    ])
    return pipeline


def train_and_evaluate(data_path: str = DATA_PATH, model_save_path: str = MODEL_PATH):
    """Main training routine."""
    print("=" * 60)
    print("Customer Support Ticket Classifier - Training Pipeline")
    print("=" * 60)

    # 1. Load and clean data
    df = load_and_validate_data(data_path)
    X = df["text"]
    y = df["category"]

    # 2. Train/test split (stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 3. Create pipeline
    model = build_pipeline()

    # 4. Train model
    print("Training model pipeline...")
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 5. Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    print("\n" + "=" * 60)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)
    print("Classification Report:")
    print(report)

    # 6. Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Trained model saved to: {model_save_path}")

    # 7. Save evaluation output to outputs/evaluation.txt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        f.write("Customer Support Ticket Classification - Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Dataset Records: {len(df)}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Testing Samples: {len(X_test)}\n")
        f.write(f"Categories: {sorted(y.unique())}\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(f"Labels: {sorted(y.unique())}\n")
        f.write(str(cm) + "\n")
    print(f"Evaluation report saved to: {EVAL_PATH}")

    return model, accuracy


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)
