# ReviewLens

An end-to-end review annotation, sentiment classification, and simulated
e-commerce storefront, built to mirror the annotation-to-model pipeline.
Takes raw Amazon product reviews through cleaning, hand-annotation, model
training, and live deployment to a demo storefront.

Full pipeline documentation: see `docs/PROJECT_DOCUMENTATION.md`
Annotation business rules and their revision history: see `GUIDELINES.md`

**Live demo:** [reviewlens-x8xk.onrender.com](https://reviewlens-x8xk.onrender.com)
(hosted on Render's free tier, the first load after a period of
inactivity may take 30–60 seconds to spin up)

## What this project demonstrates

- Data cleaning and balanced sampling from a real, messy, imbalanced dataset
- Manual data annotation using a purpose-built Flask tool, against
  predefined (and iteratively refined) business rules
- Training and evaluating a supervised text classifier on hand-labeled data
- Deploying a trained model to make live predictions in a simulated
  product context, on data verified to be unseen by the model

## Pipeline Overview

```
raw_reviews.csv (34,660 reviews, 92% positive)
        │  clean_data.py: bucket by rating, balanced sampling
        v
reviews.db (2,405 reviews: ~812 Negative / ~800 Neutral / ~800 Positive)
        │  /annotate: manual hand-labeling via Flask tool
        v
annotations.db (152 labeled reviews: sentiment, category, confidence, notes)
        │  train_model.py: TF-IDF + Logistic Regression
        v
sentiment_model.pkl + metrics.json
        │
        ├──> /dashboard: charts + confusion matrix + precision/recall over annotations.db
        └──> storefront (/, /product/<id>, /predict): live predictions,
             excluding any review already used in annotation/training
```

## Results

**123 of 152 annotated reviews** were used for training (confidence ≥ 3;
lower-confidence annotations were excluded as unreliable training signal).

| Sentiment | Usable (confidence ≥ 3) |
|---|---|
| Positive | 67 |
| Negative | 42 |
| Neutral | 14 |

**Final precision / recall / F1 per class** (held-out 20% stratified
split, n=25):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.88 | 0.88 | 0.88 | 8 |
| Neutral | 0.00 | 0.00 | 0.00 | 3 |
| Positive | 0.75 | 0.86 | 0.80 | 14 |



