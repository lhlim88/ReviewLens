"""
ReviewLens - Model training script.

Reads:  reviews.db + annotations.db (joined on review_id)
Writes: model/sentiment_model.pkl
        model/metrics.json
        model/confusion_matrix.png

Steps:
  1. Join annotated reviews (confidence >= MIN_CONFIDENCE) with review text
  2. Stratified train/test split (preserves class proportions in both sets)
  3. Train classifier (Logistic Regression, class_weight="balanced" to
     help counteract sentiment class imbalance, e.g. thin Neutral class)
  4. Evaluate: accuracy, confusion matrix (visualized), precision/recall/F1
  5. Save model with joblib
"""

import sqlite3
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display (needed for scripts/servers)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import joblib

REVIEWS_DB = "reviews.db"
ANNOTATIONS_DB = "annotations.db"
MODEL_OUT = "model/sentiment_model.pkl"
METRICS_OUT = "model/metrics.json"
CONFUSION_MATRIX_OUT = "static/confusion_matrix.png"
MIN_CONFIDENCE = 3


def load_labeled_data():
    reviews_conn = sqlite3.connect(REVIEWS_DB)
    annotations_conn = sqlite3.connect(ANNOTATIONS_DB)

    annotations = pd.read_sql(
        "SELECT review_id, sentiment, confidence FROM annotations WHERE confidence >= ?",
        annotations_conn, params=(MIN_CONFIDENCE,)
    )
    reviews = pd.read_sql("SELECT review_id, review_text FROM reviews", reviews_conn)

    reviews_conn.close()
    annotations_conn.close()

    merged = annotations.merge(reviews, on="review_id", how="inner")
    return merged[["review_text", "sentiment"]]


def plot_confusion_matrix(cm, labels, out_path):
    """Render a confusion matrix as a labeled heatmap image, saved to disk."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Write the actual count inside each cell, using dark/light text for contrast
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train():
    df = load_labeled_data()
    print(f"Training on {len(df)} labeled reviews (confidence >= {MIN_CONFIDENCE})")
    print(df["sentiment"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"], df["sentiment"], test_size=0.2, random_state=42,
        stratify=df["sentiment"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, preds)
    print("\nAccuracy:", accuracy)

    labels = sorted(df["sentiment"].unique())

    cm = confusion_matrix(y_test, preds, labels=labels)
    print("\nConfusion matrix (rows=actual, cols=predicted)")
    print("Labels order:", labels)
    print(cm)
    plot_confusion_matrix(cm, labels, CONFUSION_MATRIX_OUT)
    print(f"Confusion matrix image saved to {CONFUSION_MATRIX_OUT}")

    print("\nPrecision / Recall / F1-score per class:")
    report_text = classification_report(y_test, preds, labels=labels, zero_division=0)
    print(report_text)
    report_dict = classification_report(y_test, preds, labels=labels, zero_division=0, output_dict=True)

    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

    metrics = {
        "accuracy": round(accuracy, 4),
        "total_labeled": len(df),
        "min_confidence_used": MIN_CONFIDENCE,
        "class_counts": df["sentiment"].value_counts().to_dict(),
        "classification_report": report_dict,
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_OUT}")


if __name__ == "__main__":
    train()