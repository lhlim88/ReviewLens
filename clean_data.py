"""
ReviewLens - Data cleaning script.

Reads:  data/raw_reviews.csv
Writes: data/cleaned_reviews.csv  AND  reviews.db (table: reviews)

Pipeline:
  1. Load raw CSV, select/rename relevant columns
  2. Bucket reviews into Negative (1-2 stars) / Neutral (3 stars) / Positive (4-5 stars)
  3. Build a balanced "working pool": all Negative, capped random sample of
     Neutral and Positive (matched to Negative's natural size) to avoid the
     dataset's heavy positive-review skew
  4. Clean: drop nulls/duplicates/too-short reviews, tag language="en"
  5. Save cleaned CSV + load into reviews.db
"""

import pandas as pd
import sqlite3

RAW_PATH = "data/raw_reviews.csv"
CLEANED_PATH = "data/cleaned_reviews.csv"
DB_PATH = "reviews.db"

MIN_WORD_COUNT = 3 # to filter out reviews that are too short to be meaningfully annotated, might need more tuning.

# no NEGATIVE_SAMPLE_SIZE since all negative reviews will be used
NEUTRAL_SAMPLE_SIZE = 800
POSITIVE_SAMPLE_SIZE = 800
RANDOM_SEED = 42 


def load_raw_data():
    df = pd.read_csv(RAW_PATH, encoding="utf-8", low_memory=False)
    df = df[["name", "categories", "reviews.text", "reviews.rating", "reviews.date"]]
    df = df.rename(columns={
        "name": "product_name",
        "categories": "product_category",
        "reviews.text": "review_text",
        "reviews.rating": "rating",
        "reviews.date": "date",
    })
    return df


def bucket_by_rating(df):
    negative = df[df["rating"].isin([1, 2])]
    neutral = df[df["rating"] == 3]
    positive = df[df["rating"].isin([4, 5])]
    return negative, neutral, positive


def build_working_pool(negative, neutral, positive):
    neutral_sample = neutral.sample(n=min(NEUTRAL_SAMPLE_SIZE, len(neutral)), random_state=RANDOM_SEED)
    positive_sample = positive.sample(n=min(POSITIVE_SAMPLE_SIZE, len(positive)), random_state=RANDOM_SEED)
    pool = pd.concat([negative, neutral_sample, positive_sample], ignore_index=True)
    return pool



def clean_reviews(df):
    df = df.dropna(subset=["review_text"]) 
    df = df.drop_duplicates(subset=["review_text"])
    df = df[df["review_text"].str.split().str.len() >= MIN_WORD_COUNT] # only keep reviews with 3+ words.
    df["review_text"] = df["review_text"].str.strip()
    df["language"] = "en" # assigning a new language column
    return df


def save_outputs(df):
    df.to_csv(CLEANED_PATH, index=False, encoding="utf-8")
    conn = sqlite3.connect(DB_PATH)
    df_for_db = df.reset_index(drop=True)
    df_for_db.insert(0, "review_id", df_for_db.index + 1) #create a new 'review_id' col at the start
    df_for_db.to_sql("reviews", conn, if_exists="replace", index=False) #create table and insert values in sql
    conn.close()


def main():
    df = load_raw_data()
    negative, neutral, positive = bucket_by_rating(df)
    print(f"Available — Negative: {len(negative)}, Neutral: {len(neutral)}, Positive: {len(positive)}")

    pool = build_working_pool(negative, neutral, positive)
    print(f"Working pool before cleaning: {len(pool)}")

    cleaned = clean_reviews(pool)
    print(f"Working pool after cleaning: {len(cleaned)}")

    save_outputs(cleaned)
    print(f"Saved to {CLEANED_PATH} and {DB_PATH}")


if __name__ == "__main__":
    main()