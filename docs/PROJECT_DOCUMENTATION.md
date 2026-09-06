# Project Documentation: Review Annotation & Sentiment Classification Platform

## 1. Project Overview

**Project name:** Review Annotation & Sentiment Classification Platform (with E-Commerce Storefront Demo)

**Purpose:** An end-to-end simulation of a real-world data annotation and LLM/AI evaluation pipeline.

**One-line summary:** A Flask-based system that ingests raw product reviews, cleans them, provides a custom annotation tool to hand-label sentiment/category, trains a classifier on the labeled data, and demonstrates the trained model live on a simulated e-commerce storefront.

---

## 2. Problem Statement / Motivation

AI systems (chatbots, recommendation engines, search, translation) all rely on human-labeled data to learn what "correct" or "good" looks like. This project simulates that real-world workflow end-to-end: starting from raw, messy data, producing a clean human-annotated dataset, training a model on it, and deploying that model to make live predictions in a user-facing context.

---

## 3. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Raw Reviews    │ --> │  Data Cleaning    │ --> │    reviews.db       │
│  (CSV dataset)   │     │  (clean_data.py)  │     │   (SQLite)           │
└─────────────────┘     └──────────────────┘     └──────────┬─────────┘
                                                              │
                                                              v
                                                  ┌────────────────────┐
                                                  │  Annotation Tool    │
                                                  │  (Flask: /annotate) │
                                                  └──────────┬─────────┘
                                                              │
                                                              v
                                                  ┌────────────────────┐
                                                  │   annotations.db    │
                                                  │      (SQLite)       │
                                                  └──────────┬─────────┘
                                                              │
                                                              v
                                                  ┌────────────────────┐
                                                  │  Model Training      │
                                                  │  (train_model.py)    │
                                                  │  TF-IDF + Classifier │
                                                  └──────────┬─────────┘
                                                              │
                                                              v
                                                  ┌────────────────────┐
                                                  │ sentiment_model.pkl  │
                                                  │  (saved, loaded once)│
                                                  └───────┬──────┬─────┘
                                                          │      │
                                    ┌─────────────────────┘      └───────────────┐
                                    v                                            v
                        ┌────────────────────┐                     ┌────────────────────┐
                        │   Dashboard          │                     │   Storefront          │
                        │   (/dashboard)        │                     │  (/, /product/<id>)   │
                        │   Charts via SQL       │                     │  live predictions +   │
                        │   queries on            │                     │  /predict endpoint     │
                        │   annotations.db        │                     │  ("try it yourself")   │
                        └────────────────────┘                     └────────────────────┘
```

---

## 4. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend / web framework | Flask | Serves annotation tool, dashboard, storefront, and prediction API |
| Database | SQLite | Stores cleaned source reviews and annotation labels |
| Data manipulation | pandas | Cleaning, transformation, joining datasets |
| Machine learning | scikit-learn | TF-IDF vectorization, classifier training and evaluation |
| Model persistence | joblib | Save/load trained model without retraining |
| Frontend | HTML / CSS | Page structure and styling |
| Interactivity | JavaScript (Fetch/AJAX) | Live "try it yourself" prediction without page reload |
| Visualization | Chart.js | Dashboard charts (sentiment/category distribution, accuracy) |
| Future scalability | jieba (Chinese word segmentation) | Planned addition for Chinese-language review support |

---

## 5. Data Pipeline

### 5.1 Data Source
Public English product review dataset (e.g., Amazon Reviews subset), one product category, sampled to a manageable pool (~500-1000 reviews).

### 5.2 Cleaning Rules (`clean_data.py`)
- Remove exact duplicate reviews
- Remove empty or excessively short reviews (below a minimum word-count threshold)
- Strip HTML tags and malformed characters
- Standardize whitespace/casing where relevant
- Read/write all files with explicit UTF-8 encoding (ensures future Chinese-text compatibility)
- Tag every row with a `language` field (currently hardcoded `"en"`)

### 5.3 Output
Cleaned reviews are loaded into `reviews.db`, table `reviews`:

| Column | Type | Description |
|---|---|---|
| review_id | INTEGER (PK) | Unique identifier |
| product_name | TEXT | Product name, from the raw dataset |
| product_category | TEXT | Source category, used to map to storefront products |
| review_text | TEXT | Cleaned review content |
| rating | REAL | Original 1-5 star rating |
| date | TEXT | Original review date, if available |
| language | TEXT | Language code (`"en"`) |

---

## 6. Annotation System

### 6.1 Annotation Guidelines (Business Rules)

**Sentiment labels:**
- **Positive** — reviewer is satisfied overall, even if minor complaints exist
- **Negative** — reviewer is dissatisfied overall, even if some positives are mentioned
- **Neutral** — mixed, purely factual, or no clear sentiment expressed

**Category labels:**
- **Product Quality** — durability, defects, performance, including
  software bugs/crashes/lag (genuine malfunctions)
- **Shipping Issue** — delivery time, packaging, damage in transit
- **Price/Value** — pricing/value for money, OR general positive
  mentions of features, capabilities, or suitability for a use case
  (renamed and broadened from an original "Price Complaint" — see
  `GUIDELINES.md` for the revision history)
- **Customer Service** — support interactions, returns, refunds
- **Other** — anything not covered above, including ecosystem/policy
  complaints unrelated to a malfunction, and vague/incoherent reviews

**Confidence score:** 1 (low confidence / ambiguous) to 5 (very confident)

**Notes field:** Optional; used primarily for Negative or ambiguous reviews to record reasoning (e.g., sarcasm, mixed signals, unclear phrasing).

### 6.2 Annotation Tool (Flask `/annotate`)
- Displays one un-annotated review at a time, pulled from `reviews.db`,
  chosen at random (`ORDER BY RANDOM()`) rather than sequentially
- Shows star rating (kept for annotation speed); deliberately does NOT
  show product name (removed mid-batch after observing it caused
  price/brand anchoring bias — see `GUIDELINES.md`)
- Form fields: sentiment (dropdown), category (dropdown), confidence
  (dropdown, 1–5), notes (text area, optional)
- On submit: writes a row to `annotations.db`, then loads the next
  random un-annotated review
- Two-pass annotation workflow: first pass captures verdict/category/confidence quickly across the full batch; second pass revisits flagged (e.g., Negative or low-confidence) items to add notes

### 6.3 Annotation Storage Schema (`annotations.db`, table `annotations`)

| Column | Type | Description |
|---|---|---|
| annotation_id | INTEGER (PK) | Unique identifier |
| review_id | INTEGER (FK) | Links to `reviews.review_id` |
| sentiment | TEXT | Positive / Negative / Neutral |
| category | TEXT | One of the defined category labels |
| confidence | INTEGER | 1–5 |
| notes | TEXT | Optional free text |
| timestamp | TEXT | When the annotation was submitted |


## 7. Model Training & Evaluation

### 7.1 Approach
- Join `annotations.db` and `reviews.db` on `review_id` to produce a labeled dataset, filtered to confidence ≥ 3 (123 of 152 annotations used)
- Stratified split into training (~80%) and held-out test (~20%) sets, preserving class proportions across both
- Text vectorization via TF-IDF, with `ngram_range=(1,2)` (unigrams + bigrams, to capture negation like "not good") and `stop_words="english"`, both added iteratively and validated with controlled before/after comparisons.
- Classifier: Logistic Regression (`class_weight="balanced"` to help counteract sentiment class imbalance, particularly the thin Neutral class). 

### 7.2 Evaluation Metrics
- Accuracy on held-out test set (final: 76%, up from a 56% unigram-only baseline)
- Confusion matrix across sentiment classes, rendered as a labeled heatmap image (`plot_confusion_matrix()`, saved to `static/confusion_matrix.png`)
- Precision / recall / F1-score per class (`classification_report`), saved into `model/metrics.json` alongside accuracy and class counts
- Training-vs-test accuracy comparison, used to check for overfitting (found: 100% training accuracy vs 76% test accuracy — a real overfitting signal, attributed to having more vocabulary features than training rows)

### 7.3 Model Persistence
Trained model serialized with `joblib` to `model/sentiment_model.pkl`, loaded once at Flask application startup and reused for all subsequent predictions. Metrics are separately persisted to `model/metrics.json` on each training run, so the dashboard always reflects the most recent training result without needing to retrain to display it.

---

## 8. Dashboard

**Route:** `/dashboard`

**Data sources:** Live SQL queries against `annotations.db` (`GROUP BY sentiment`, `GROUP BY category`, `GROUP BY confidence`), plus `model/metrics.json` (written by `train_model.py`)

**Layout:**
- Headline stat cards: total reviews annotated, model accuracy, count used for training
- Sentiment distribution (doughnut chart, color-coded to match storefront badge colors)
- Category breakdown (horizontal bar chart)
- Confidence score distribution (bar chart)
- Confusion matrix (static image, `static/confusion_matrix.png`, generated by `train_model.py`)
- Precision / recall / F1 table, per class, from `metrics.json`'s `classification_report`

If no model has been trained yet (`model/metrics.json` absent), the accuracy stat card and confusion matrix/precision-recall sections are omitted, with a note prompting the user to run `train_model.py`.

---

## 9. E-Commerce Storefront Simulation

### 9.1 Purpose
Demonstrates the trained model in a realistic, user-facing context, reviews are classified live, not just evaluated in isolation.

### 9.2 Pages & Routes

| Route | Description |
|---|---|
| `/` | Product grid — fake products, each mapped to a `product_category` |
| `/product/<id>` | Product detail page; pulls that product's reviews from `reviews.db` and runs live model predictions, displaying each with a sentiment badge |
| `/predict` (POST) | Accepts arbitrary review text, returns JSON prediction `{sentiment, confidence}`; powers the "try it yourself" interactive box |

### 9.3 "Try It Yourself" Feature
A text input on the storefront where a visitor types a review and receives an instant AI-generated sentiment prediction via a JavaScript `fetch` call to `/predict`, with the result rendered on the page without a full reload.

### 9.4 Data Source for Storefront Reviews
Storefront product pages display reviews drawn from the full cleaned pool in `reviews.db` — including the portion never manually annotated, meaning the model is making genuine live predictions on unseen data, in addition to its formal held-out test set evaluation from Section 7.

---


## 11. Project Folder Structure

```
reviewlens/
│
├── data/
│   ├── raw_reviews.csv
│   └── cleaned_reviews.csv
│
├── clean_data.py
├── check_progress.py
├── reviews.db
├── annotations.db
│
├── train_model.py
├── model/
│   ├── sentiment_model.pkl
│   └── metrics.json
│
├── app.py
├── templates/
│   ├── annotate.html
│   ├── dashboard.html
│   ├── store_home.html
│   └── product_detail.html
│
├── static/
│   ├── style.css
│   ├── predict.js
│   └── confusion_matrix.png       (generated by train_model.py)
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md
│
├── GUIDELINES.md
└── README.md
```

## 12. Known Limitations / Deliberate Tradeoffs
 
These were identified through actually doing the annotation and modeling work, not assumed in advance.
 
- **Neutral class is underrepresented** in training data (14 usable examples vs. 67 Positive), directly limiting the model's ability to recognize Neutral sentiment. A larger or more deliberately targeted annotation batch would address this.
- **Vocabulary coverage is limited** to words present in the 123 training reviews — a larger dataset would directly expand this.
- **Model overfits** (100% train vs. 76% test accuracy) — a direct consequence of having more vocabulary features (3,669) than training rows (~98); see Section 7.5.
- **Star rating was shown during annotation** for labeling speed, despite carrying some anchoring-bias risk (labels could be influenced by the numeric rating rather than the text alone). A stricter pipeline would hide it.
- **Product identity/price was hidden during annotation** after noticing, mid-annotation, that seeing the product influenced sentiment judgments via implicit price/brand associations (e.g., reading identical review text as more positive for a cheap product than an expensive one). Fixed partway through the annotation batch — see `GUIDELINES.md` for details.
- **Annotation guidelines were refined iteratively** as edge cases arose (stacked-complaint reviews, multi-topic reviews, price-vs-quality category boundaries, confidence miscalibration). Earlier annotations may be marginally less consistent with the final guidelines than later ones — a known tradeoff of solo, real-time annotation without a separate guideline-pilot phase.
- **No cross-validation** — a single stratified train/test split was used rather than k-fold cross-validation, an appropriate tradeoff at this dataset size (~123 rows) and project scope; the 76% accuracy figure should be read as indicative, not precise.
- **Class imbalance was addressed via feature engineering, not more data, for Negative/Positive** — bigrams and stopword removal improved performance on the two better-represented classes, but this approach had no effect on Neutral, confirming its failure is specifically a data-volume problem that feature engineering cannot solve.


## 14. Key Learnings
 
**Annotation**
- Anchoring bias is real and self-observable: sentiment judgments were caught drifting based on implied product price/brand, not just the review text, fixed by hiding `product_name` mid-batch.
- Confidence needed a precisely scoped definition ("certainty in sentiment specifically," since only sentiment trains the model) to stay consistent, and was still caught drifting from that definition later, showing how easily such rules erode without active discipline.
- Category definitions evolved under real examples (Price Complaint → Price/Value → broadened definition) as recurring patterns emerged that didn't fit the original five categories.

**Modeling & results**
- Only hand-labeled rows are usable for training, the ~2,250 un-annotated reviews remain structurally invisible to the model regardless of volume.
- Confidence-filtering (≥3) reduced 152 labeled rows to 123 usable ones at the cost of uncertain annotations
- Two targeted, evidence-tested improvements (bigrams, then stopword removal) raised accuracy from 56% → 64% → 76%, each validated with a controlled before/after comparison rather than assumed. Both improved Negative and Positive performance but had zero effect on Neutral, a strong evidence that Neutral's failure is a data-volume problem that feature engineering cannot fix, no matter how it's tuned.
- Comparing training vs. test accuracy (100% vs. 76%) surfaced a genuine overfitting signal, traced to having more vocabulary features (3,669) than training rows (~98) — a direct, quantifiable illustration of why feature-to-example ratio matters, not just raw accuracy.
- Three distinct, verified root causes were found behind remaining weak predictions on genuinely unseen text: (1) missing vocabulary, (2) TF-IDF's exact-string matching failing on typos, and (3) no sarcasm/irony handling. Each reflects a different underlying limitation — (1) is fixable with more data, (2) and (3) are structural limitations of TF-IDF and word-frequency models generally.
- The storefront's live predictions were verified to run only on reviews never seen during annotation or training (`product_detail()` excludes any `review_id` present in `annotations.db`).
