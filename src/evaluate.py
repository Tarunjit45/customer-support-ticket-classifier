"""
Model Evaluation Script for Customer Support Ticket Classification.

This script:
1. Loads the saved model from disk.
2. Loads the dataset and applies the reproducible test split.
3. Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
4. Displays the results and saves the report to outputs/evaluation.txt.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ticket_classifier.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EVAL_PATH = os.path.join(OUTPUT_DIR, "evaluation.txt")


def evaluate_model():
    """Evaluate saved model on test data and output metrics."""
    # 1. Validation checks
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please train the model first by running: python src/train.py"
        )

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'.")

    # 2. Load model
    print(f"Loading trained model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # 3. Load dataset
    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'category' columns.")

    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["category"] != "")].drop_duplicates(subset=["text"])

    X = df["text"]
    y = df["category"]

    # 4. Stratified test split (identical to training split)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Predict on test set
    y_pred = model.predict(X_test)

    # 6. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 7. Display output
    print("\n" + "=" * 60)
    print("           MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Evaluated on {len(X_test)} unseen test samples across {len(labels)} categories.")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(report)

    print("Confusion Matrix:")
    print(f"Labels: {labels}")
    print(cm)
    print("=" * 60)

    # 8. Save report to outputs/evaluation.txt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        f.write("Customer Support Ticket Classification - Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Dataset Records: {len(df)}\n")
        f.write(f"Testing Samples: {len(X_test)}\n")
        f.write(f"Categories: {labels}\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(f"Labels: {labels}\n")
        f.write(str(cm) + "\n")

    print(f"Report saved to: {EVAL_PATH}\n")
    return accuracy, report, cm


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
