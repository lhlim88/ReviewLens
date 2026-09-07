"""
ReviewLens - Main Flask application.

Routes:
  GET  /               -> storefront home (product grid)         
  GET  /product/<id>   -> product detail page                    
  POST /predict          -> JSON prediction API                   
  GET  /annotate          -> show a random un-annotated review
  POST /annotate           -> save submitted annotation, show next
  GET  /dashboard          -> charts summarizing annotations.db    ,
"""

from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import os
import joblib
from datetime import datetime

app = Flask(__name__)

REVIEWS_DB = "reviews.db"
ANNOTATIONS_DB = "annotations.db"
METRICS_FILE = "model/metrics.json"
MODEL_FILE = "model/sentiment_model.pkl"
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"

# Fake storefront products, mapped to real product_name substrings in reviews.db
PRODUCTS = [
    {"id": 1, "name": "Fire 7 Tablet", "price": "$49.99", "match": "Fire Tablet, 7 Display"},
    {"id": 2, "name": "Echo Speaker", "price": "$89.99", "match": "Echo (White)"},
    {"id": 3, "name": "Kindle Paperwhite", "price": "$119.99", "match": "Kindle Paperwhite"},
    {"id": 4, "name": "Fire HD 8 Tablet", "price": "$79.99", "match": "All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB"},
    {"id": 5, "name": "Fire TV", "price": "$39.99", "match": "Amazon Fire Tv"},
    {"id": 6, "name": "Fire Kids Tablet", "price": "$99.99", "match": "Fire Kids Edition Tablet"},
]

# Load the trained model once at startup, kept "warm" in memory for all requests
_model_bundle = None
if os.path.exists(MODEL_FILE):
    _model_bundle = joblib.load(MODEL_FILE)


def predict_sentiment(text):
    """Run the loaded model on a piece of text. Returns (label, confidence) or (None, None) if no model loaded."""
    if _model_bundle is None:
        return None, None
    vectorizer = _model_bundle["vectorizer"]
    classifier = _model_bundle["classifier"]
    vec = vectorizer.transform([text])
    label = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]
    confidence = max(proba)
    return label, round(float(confidence), 3)


def init_annotations_db():
    conn = sqlite3.connect(ANNOTATIONS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            sentiment TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            notes TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_random_unannotated_review():
    reviews_conn = sqlite3.connect(REVIEWS_DB)
    annotations_conn = sqlite3.connect(ANNOTATIONS_DB)

    already_annotated = [row[0] for row in annotations_conn.execute("SELECT review_id FROM annotations")]
    annotations_conn.close()

    if already_annotated:
        placeholders = ",".join("?" * len(already_annotated))
        query = f"""SELECT review_id, review_text, product_name, rating
                     FROM reviews WHERE review_id NOT IN ({placeholders})
                     ORDER BY RANDOM() LIMIT 1"""
        row = reviews_conn.execute(query, already_annotated).fetchone()
    else:
        query = """SELECT review_id, review_text, product_name, rating
                    FROM reviews ORDER BY RANDOM() LIMIT 1"""
        row = reviews_conn.execute(query).fetchone()

    reviews_conn.close()
    return row


def get_annotation_count():
    conn = sqlite3.connect(ANNOTATIONS_DB)
    count = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    conn.close()
    return count


@app.route("/annotate", methods=["GET", "POST"])
def annotate():
    if request.method == "POST":
        if not DEMO_MODE:
            review_id = request.form["review_id"]
            sentiment = request.form["sentiment"]
            category = request.form["category"]
            confidence = request.form["confidence"]
            notes = request.form.get("notes", "")

            conn = sqlite3.connect(ANNOTATIONS_DB)
            conn.execute(
                """INSERT INTO annotations
                (review_id, sentiment, category, confidence, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (review_id, sentiment, category, confidence, notes, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()

    review = get_random_unannotated_review()
    count = get_annotation_count()

    if review is None:
        return f"All reviews annotated! Total labeled: {count}"

    return render_template(
        "annotate.html",
        review_id=review[0],
        review_text=review[1],
        product_name=review[2],
        rating=review[3],
        annotated_count=count,
    )


@app.route("/")
def store_home():
    return render_template("store_home.html", products=PRODUCTS)


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if product is None:
        return "Product not found", 404

    annotations_conn = sqlite3.connect(ANNOTATIONS_DB)
    annotated_ids = [row[0] for row in annotations_conn.execute("SELECT review_id FROM annotations")]
    annotations_conn.close()

    conn = sqlite3.connect(REVIEWS_DB)
    if annotated_ids:
        placeholders = ",".join("?" * len(annotated_ids))
        query = f"""SELECT review_text FROM reviews
                     WHERE product_name LIKE ? AND review_id NOT IN ({placeholders})
                     LIMIT 8"""
        rows = conn.execute(query, [f"%{product['match']}%"] + annotated_ids).fetchall()
    else:
        rows = conn.execute(
            "SELECT review_text FROM reviews WHERE product_name LIKE ? LIMIT 8",
            (f"%{product['match']}%",)
        ).fetchall()
    conn.close()

    reviews_with_predictions = []
    for (text,) in rows:
        label, confidence = predict_sentiment(text)
        reviews_with_predictions.append({
            "text": text,
            "sentiment": label,
            "confidence": confidence,
        })

    return render_template(
        "product_detail.html",
        product=product,
        reviews=reviews_with_predictions,
        model_loaded=_model_bundle is not None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip() if data else ""

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if _model_bundle is None:
        return jsonify({"error": "Model not loaded - run train_model.py first"}), 503

    label, confidence = predict_sentiment(text)
    return jsonify({"sentiment": label, "confidence": confidence})


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(ANNOTATIONS_DB)

    sentiment_counts = dict(conn.execute(
        "SELECT sentiment, COUNT(*) FROM annotations GROUP BY sentiment"
    ).fetchall())

    category_counts = dict(conn.execute(
        "SELECT category, COUNT(*) FROM annotations GROUP BY category"
    ).fetchall())

    confidence_counts = dict(conn.execute(
        "SELECT confidence, COUNT(*) FROM annotations GROUP BY confidence"
    ).fetchall())

    total = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    conn.close()

    metrics = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            metrics = json.load(f)

    return render_template(
        "dashboard.html",
        total=total,
        sentiment_labels=list(sentiment_counts.keys()),
        sentiment_values=list(sentiment_counts.values()),
        category_labels=list(category_counts.keys()),
        category_values=list(category_counts.values()),
        confidence_labels=list(confidence_counts.keys()),
        confidence_values=list(confidence_counts.values()),
        metrics=metrics,
    )


if __name__ == "__main__":
    init_annotations_db()
    app.run(debug=True, port=5001)