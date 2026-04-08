import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE  = PROCESSED_DIR / "dataset_preprocessed.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_preprocessed.csv"
# overwrites in place

df = pd.read_csv(INPUT_FILE)

before = len(df)

# drop exact duplicate emails
df = df.drop_duplicates(subset=["text_combined"])

# drop near-empty bodies (no training signal)
df = df[df["body_length_words"] >= 5]

df.to_csv(OUTPUT_FILE, index=False)

print(f"Rows before: {before}")
print(f"Rows after:  {len(df)}")
print(f"Removed:     {before - len(df)}")
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nSource counts:")
print(df["source"].value_counts())
