"""
Train and evaluate an explainable fake job posting classifier.
Dataset: Kaggle Real or Fake Fake Job Posting Prediction
Expected CSV filename: fake_job_postings.csv
Target column: fraudulent (0 = real, 1 = fraudulent)
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "fake_job_postings.csv"
MODEL_PATH = ROOT / "fraud_detection_model.joblib"
METRICS_PATH = ROOT / "metrics.json"
CM_PATH = ROOT / "confusion_matrix.png"
REPORT_PATH = ROOT / "classification_report.txt"

TEXT_COLUMNS = [
    "title", "company_profile", "description", "requirements", "benefits",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "location", "department"
]

def combine_text(frame):
    """Combine available job-posting fields into one text string."""
    available = [col for col in TEXT_COLUMNS if col in frame.columns]
    if not available:
        raise ValueError("No expected text/metadata columns found in the CSV.")
    cleaned = frame[available].fillna("").astype(str)
    return cleaned.agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Kaggle dataset and place its CSV here as fake_job_postings.csv."
        )

    data = pd.read_csv(DATA_PATH)
    if "fraudulent" not in data.columns:
        raise ValueError("The CSV must contain a 'fraudulent' target column.")

    data = data.dropna(subset=["fraudulent"]).copy()
    data["fraudulent"] = pd.to_numeric(data["fraudulent"], errors="coerce")
    data = data.dropna(subset=["fraudulent"])
    data["fraudulent"] = data["fraudulent"].astype(int)
    data = data[data["fraudulent"].isin([0, 1])]

    X = combine_text(data)
    y = data["fraudulent"]

    if y.nunique() != 2:
        raise ValueError("The target must contain both classes: 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=50000,
            ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ))
    ])

    print("Training Logistic Regression with TF-IDF...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_fraud": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_fraud": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_fraud": float(f1_score(y_test, predictions, zero_division=0)),
        "note": "Metrics are from this dataset split; they do not guarantee real-world performance."
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(classification_report(
        y_test, predictions, target_names=["Legitimate (0)", "Potential fraud (1)"],
        zero_division=0
    ), encoding="utf-8")

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions),
        display_labels=["Legitimate", "Potential fraud"]
    )
    display.plot(values_format="d")
    plt.title("Fake Job Posting Detection — Test Set")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=160)
    plt.close()

    print("\nEvaluation results")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved metrics: {METRICS_PATH.name}")
    print(f"Saved confusion matrix: {CM_PATH.name}")
    print(f"Saved report: {REPORT_PATH.name}")

if __name__ == "__main__":
    main()
