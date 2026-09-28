# Customer Support Ticket Classification System

An end-to-end Machine Learning system for automated customer support ticket classification, built for an AI/ML technical assignment.

> 📖 **Looking for the deep dive?** Read the complete [Comprehensive Technical & Architecture Guide](DOCUMENTATION.md) for plain-English explanations, mathematical details, and fresher interview Q&A.

---

## Overview

In modern customer service operations, incoming support tickets must be quickly routed to the correct department (e.g., Billing, Technical Support, Delivery) to reduce response times and improve customer satisfaction. This project implements a lightweight, reproducible machine learning pipeline that learns from historical ticket data and predicts the category of new, unseen customer support requests.

---

## Objective

The objective is to take raw customer support ticket text such as:

> *"I was charged twice for my subscription this month."*

and predict the appropriate support category:

> **Billing** (with confidence score)

The system manages the entire lifecycle:
$$\text{Data Ingestion} \rightarrow \text{Data Cleaning} \rightarrow \text{TF-IDF Vectorization} \rightarrow \text{Logistic Regression Training} \rightarrow \text{Evaluation} \rightarrow \text{Model Persistence} \rightarrow \text{Inference}$$

---

## Features

- **Automated Data Validation & Cleaning:** Drops missing values, removes empty records, strips whitespace, and removes duplicate tickets.
- **TF-IDF Feature Extraction:** Converts raw text into unigram and bigram numerical vectors with English stop-word filtering.
- **Classical ML Classifier:** Employs Logistic Regression with probability estimation for interpretable confidence scores.
- **Scikit-Learn Pipeline:** Bundles vectorizer and classifier into a single serialized pipeline to prevent data leakage.
- **Model Evaluation:** Computes Accuracy, Precision, Recall, F1-Score (macro/weighted), and Confusion Matrix on an unseen 20% stratified test split.
- **Flexible Inference:** Predict via interactive terminal prompt or direct command-line arguments.
- **Dual Interface:** Supports standalone modular scripts (`train.py`, `evaluate.py`, `predict.py`) and an all-in-one CLI entry point (`main.py`).

---

## Technologies

- **Python 3.12**
- **pandas** (Data loading, validation, and manipulation)
- **NumPy** (Numerical operations)
- **scikit-learn** (TF-IDF vectorizer, Logistic Regression classifier, train/test splitting, evaluation metrics)
- **joblib** (Model serialization and loading)

---

## Project Structure

```text
customer-support-ticket-classifier/
│
├── data/
│   └── tickets.csv                  # Historical customer support tickets dataset
│
├── models/
│   └── ticket_classifier.joblib     # Serialized trained scikit-learn pipeline
│
├── outputs/
│   └── evaluation.txt               # Evaluation metrics and confusion matrix output
│
├── src/
│   ├── train.py                     # Training script: loads data, fits pipeline, saves model
│   ├── evaluate.py                  # Evaluation script: tests model on test split, logs metrics
│   └── predict.py                   # Prediction script: interactive and CLI inference
│
├── main.py                          # Unified CLI entry point with interactive menu
├── requirements.txt                 # Minimal project dependencies
├── .gitignore                       # Git ignore rules for Python artifacts and caches
└── README.md                        # Project documentation and guide
```

---

## Dataset

- **Source:** A curated demonstration dataset created specifically for this assignment, representing realistic customer support queries.
- **Total Records:** 150 tickets
- **Columns:**
  - `ticket_id`: Unique identifier (e.g., `TCK-1001`)
  - `text`: Customer inquiry or problem statement
  - `category`: Ground truth support category
- **Categories (30 records each, balanced):**
  1. `Account`: Password resets, 2FA issues, account deletion, profile updates
  2. `Billing`: Unexpected charges, invoice requests, payment failures, subscription fees
  3. `Delivery`: Delayed shipping, lost packages, tracking issues, wrong delivery address
  4. `Refund`: Refund requests, return processing, money-back guarantees
  5. `Technical Support`: Application crashes, server errors, UI/UX bugs, sync failures

---

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd customer-support-ticket-classifier
```

### 2. Create and activate a virtual environment

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

You can run the project either through the individual module scripts or through `main.py`.

### Option A: Using `main.py`

- **Interactive Menu:**
  ```bash
  python main.py
  ```
- **Train the model:**
  ```bash
  python main.py --train
  ```
- **Evaluate the model:**
  ```bash
  python main.py --evaluate
  ```
- **Predict a ticket:**
  ```bash
  python main.py --predict "I was charged twice for my order"
  ```

---

### Option B: Using Individual Scripts

#### How to Train
```bash
python src/train.py
```
This loads `data/tickets.csv`, splits 80% train / 20% test (stratified), fits the pipeline, outputs performance metrics, and saves the trained model to `models/ticket_classifier.joblib`.

#### How to Evaluate
```bash
python src/evaluate.py
```
This evaluates the saved model on the test split and updates `outputs/evaluation.txt`.

#### How to Predict
- **Via CLI Argument:**
  ```bash
  python src/predict.py --text "My package has not arrived and tracking is stuck."
  ```
- **Interactive Mode:**
  ```bash
  python src/predict.py
  ```

---

## Actual Execution & Test Results

The following metrics were obtained by executing `src/train.py` and `src/evaluate.py` on the 20% test split (30 unseen samples):

```text
============================================================
           MODEL EVALUATION SUMMARY
============================================================
Evaluated on 30 unseen test samples across 5 categories.
Overall Accuracy: 0.8667 (86.67%)

Classification Report:
                   precision    recall  f1-score   support

          Account     0.8571    1.0000    0.9231         6
          Billing     0.8000    0.6667    0.7273         6
         Delivery     1.0000    0.8333    0.9091         6
           Refund     0.8571    1.0000    0.9231         6
Technical Support     0.8333    0.8333    0.8333         6

         accuracy                         0.8667        30
        macro avg     0.8695    0.8667    0.8632        30
     weighted avg     0.8695    0.8667    0.8632        30

Confusion Matrix:
Labels: ['Account', 'Billing', 'Delivery', 'Refund', 'Technical Support']
[[6 0 0 0 0]
 [0 4 0 1 1]
 [1 0 5 0 0]
 [0 0 0 6 0]
 [0 1 0 0 5]]
```

### Real Prediction Examples

| Input Ticket | Predicted Category | Confidence |
| :--- | :--- | :--- |
| `"I cannot log into my account because my password reset link is broken."` | **Account** | 0.52 |
| `"I was charged twice for my subscription this morning, please fix this."` | **Billing** | 0.35 |
| `"My package has not arrived and the tracking number shows delayed."` | **Delivery** | 0.45 |
| `"The application keeps crashing with a 500 error when I open the dashboard."` | **Technical Support** | 0.42 |
| `"I want a full refund because the item arrived broken and defective."` | **Refund** | 0.36 |

---

## Model Architecture

The classification system utilizes a standard, production-ready scikit-learn `Pipeline`:

1. **TF-IDF Vectorizer (`TfidfVectorizer`):**
   - Converts raw textual tickets into numerical sparse feature vectors.
   - Computes Term Frequency-Inverse Document Frequency to emphasize domain-specific keywords (e.g., "charge", "refund", "crash") while penalizing common stopwords.
   - Configured with `ngram_range=(1, 2)` to capture both single words and two-word phrases (e.g., "charged twice", "password reset").
2. **Logistic Regression (`LogisticRegression`):**
   - Fits linear decision boundaries with multi-class one-vs-rest formulation.
   - Produces well-calibrated class probability distributions via `predict_proba`.
   - Fast, highly interpretable, and resilient against overfitting on small text datasets.

---

## Limitations

- **Dataset Size:** The dataset contains 150 sample tickets. While sufficient for demonstrating a working pipeline, production systems require thousands of records.
- **Lexical Overlap:** Ambiguous tickets containing overlapping terms (e.g., "I want a refund for the billing charge") may trigger borderline probabilities between `Billing` and `Refund`.
- **Out-of-Vocabulary (OOV) Handling:** Slang, heavy misspellings, or languages other than English may reduce classification confidence.
- **Confidence Scores:** Confidence values reflect soft-max probabilities across 5 categories rather than an absolute guarantee of correctness.

---

## Future Improvements

- Incorporate lemmatization and spelling correction during preprocessing.
- Expand dataset size with diverse domain-specific edge cases.
- Perform hyperparameter tuning (e.g., regularization parameter $C$, n-gram ranges) using Cross-Validation.
- Implement an automated fallback/escalation queue for tickets with confidence scores below a defined threshold (e.g., $< 0.30$).
