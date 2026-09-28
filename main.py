"""
Customer Support Ticket Classification System - Main Entry Point.

Usage:
  Interactive menu:
    python main.py

  Direct CLI commands:
    python main.py --train
    python main.py --evaluate
    python main.py --predict "I was charged twice for my subscription."
"""

import sys
import argparse
from src.train import train_and_evaluate
from src.evaluate import evaluate_model
from src.predict import predict_ticket, load_classifier


def show_menu():
    """Display interactive menu."""
    while True:
        print("\n" + "=" * 52)
        print("  Customer Support Ticket Classification System")
        print("=" * 52)
        print("1. Train Model")
        print("2. Evaluate Model")
        print("3. Predict Ticket Category")
        print("4. Exit")
        print("=" * 52)

        try:
            choice = input("Select an option (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if choice == "1":
            try:
                train_and_evaluate()
            except Exception as e:
                print(f"Error during training: {e}")
        elif choice == "2":
            try:
                evaluate_model()
            except Exception as e:
                print(f"Error during evaluation: {e}")
        elif choice == "3":
            try:
                ticket_text = input("\nEnter customer support ticket: ").strip()
                if not ticket_text:
                    print("Error: Ticket text cannot be empty.")
                    continue
                category, conf = predict_ticket(ticket_text)
                print("-" * 40)
                print(f"Predicted category: {category}")
                if conf > 0:
                    print(f"Confidence: {conf:.2f}")
                print("-" * 40)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "4":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, or 4.")


def main():
    parser = argparse.ArgumentParser(
        description="Customer Support Ticket Classification System"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the classification model pipeline and save it to models/"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the saved model on the test split and output metrics"
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Classify a single customer support ticket text"
    )

    args = parser.parse_args()

    # Route based on CLI arguments
    if args.train:
        train_and_evaluate()
    elif args.evaluate:
        evaluate_model()
    elif args.predict:
        category, conf = predict_ticket(args.predict)
        print("-" * 40)
        print(f"Ticket: \"{args.predict}\"")
        print(f"Predicted category: {category}")
        if conf > 0:
            print(f"Confidence: {conf:.2f}")
        print("-" * 40)
    else:
        # Default to interactive menu
        show_menu()


if __name__ == "__main__":
    main()
