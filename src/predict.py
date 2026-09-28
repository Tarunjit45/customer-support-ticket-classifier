"""
Prediction Script for Customer Support Ticket Classification.

This script:
1. Loads the saved classification pipeline.
2. Accepts a new customer support ticket text (via CLI argument or interactive prompt).
3. Predicts the appropriate support category.
4. Outputs the predicted category along with model confidence probability.
"""

import os
import sys
import argparse
import joblib

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ticket_classifier.joblib")


def load_classifier(model_path: str = MODEL_PATH):
    """Load the trained model pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by running: python src/train.py"
        )
    return joblib.load(model_path)


def predict_ticket(text: str, model=None) -> tuple[str, float]:
    """Predict category and confidence for a given ticket text.

    Args:
        text: The ticket content string.
        model: Optional pre-loaded pipeline model.

    Returns:
        tuple of (predicted_category, confidence_score)
    """
    cleaned_text = text.strip() if text else ""
    if not cleaned_text:
        raise ValueError("Ticket text cannot be empty.")

    if model is None:
        model = load_classifier()

    # Predict category
    predicted_category = model.predict([cleaned_text])[0]

    # Predict probability confidence if classifier supports predict_proba
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(max(probabilities))

    return predicted_category, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict the category of a customer support ticket."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Customer support ticket text enclosed in quotes.",
        default=None
    )
    args = parser.parse_args()

    try:
        model = load_classifier()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # If text is provided as argument
    if args.text:
        try:
            category, conf = predict_ticket(args.text, model=model)
            print("-" * 40)
            print(f"Ticket: \"{args.text}\"")
            print(f"Predicted category: {category}")
            if conf > 0:
                print(f"Confidence: {conf:.2f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Interactive mode
    print("=" * 50)
    print("Customer Support Ticket Classification - Predictor")
    print("Type a ticket description to classify.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nEnter customer support ticket: ").strip()
            if not user_input:
                print("Please enter non-empty text.")
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting.")
                break

            category, conf = predict_ticket(user_input, model=model)
            print(f"Predicted category: {category}")
            if conf > 0:
                print(f"Confidence: {conf:.2f}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
