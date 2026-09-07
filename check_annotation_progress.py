"""
ReviewLens - Quick annotation progress checker.

"""

import sqlite3

conn = sqlite3.connect("annotations.db")

total = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
print(f"Total annotated: {total}\n")

print("By sentiment:")
for row in conn.execute("SELECT sentiment, COUNT(*) FROM annotations GROUP BY sentiment ORDER BY COUNT(*) DESC"):
    print(f"  {row[0]:10s} {row[1]}")

print("\nBy category:")
for row in conn.execute("SELECT category, COUNT(*) FROM annotations GROUP BY category ORDER BY COUNT(*) DESC"):
    print(f"  {row[0]:15s} {row[1]}")

print("\nBy confidence level:")
for row in conn.execute("SELECT confidence, COUNT(*) FROM annotations GROUP BY confidence ORDER BY confidence"):
    print(f"  confidence {row[0]}: {row[1]}")

usable = conn.execute("SELECT COUNT(*) FROM annotations WHERE confidence >= 3").fetchone()[0]
print(f"\nUsable for training (confidence >= 3): {usable} out of {total}")

print("\nUsable rows by sentiment (what train_model.py will actually see):")
for row in conn.execute("SELECT sentiment, COUNT(*) FROM annotations WHERE confidence >= 3 GROUP BY sentiment ORDER BY COUNT(*) DESC"):
    print(f"  {row[0]:10s} {row[1]}")

conn.close()