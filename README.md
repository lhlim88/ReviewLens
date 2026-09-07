# ReviewLens

An end-to-end review annotation, sentiment classification, and simulated
e-commerce storefront, built to mirror the annotation-to-model pipeline.
Takes raw Amazon product reviews through cleaning, hand-annotation, model
training, and live deployment to a demo storefront.

Full pipeline documentation: see `docs/PROJECT_DOCUMENTATION.md`
Annotation business rules and their revision history: see `GUIDELINES.md`


## Try it yourself
 **Live Demo:**
 **Click link here:**
[reviewlens-x8xk.onrender.com](https://reviewlens-x8xk.onrender.com)
(hosted on Render's free tier, the first load after a period of
inactivity may take 30–60 seconds to spin up)
 
**`/annotate`** the actual annotation tool used to hand-label the 152
reviews. This is pure human labeling based on the annotation rules in `GUIDELINES.md`.
Note: this route runs in demo mode on the public deployment, so your submissions are not saved.

**How to use it:**
1. Read the review text and its star rating shown on the page
2. Pick a **Sentiment** (Positive / Negative / Neutral) based on the text
3. Pick a **Category** (Product Quality / Shipping Issue / Price-Value /
   Customer Service / Other) — see `GUIDELINES.md` for definitions
4. Set a **Confidence** score (1 = unsure, 5 = very confident)
5. Optionally add a note explaining your reasoning, especially for
   ambiguous cases.
6. Click **Submit & Next** to move on to the next review
 
**"Try it yourself" box** (on any product page) sends your typed text
to the *trained model* for a live sentiment prediction. This one has a
real limitation worth knowing before you try it: the model's vocabulary
comes entirely from the 123 training reviews, so it will confidently
misjudge text containing words it never saw during training (e.g.,
"horrible," "hate," or misspellings like "dissappointed").

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

## Design Philosophy: Why Build the Model From Scratch
 
**No pre-built sentiment lexicons** (e.g., VADER, SentiWordNet). These
are ready-made dictionaries mapping words directly to sentiment scores,
built from millions of examples and swapping one in would have sidestepped
the vocabulary-coverage problem entirely, with zero hand-annotation
required. I did not use these libraries as I wanted understand what is really 
going on under the hood of the annotation and model-training workflow.

**No LLM APIs** (e.g., Anthropic, OpenAI) for the sentiment classifier
itself. A modern LLM would likely classify sentiment more accurately,
and with far less setup, than a TF-IDF + Logistic Regression model
trained on 123 examples. Similar to the previous parahraph, the deliberate choice was to build the classical pipeline from scratch and using them would have substituted a black box for the exact process I wanted to learn. 
 


