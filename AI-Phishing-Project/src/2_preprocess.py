import re
import pandas as pd
from pathlib import Path

#========Path configuration===============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "dataset_final.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_preprocessed.csv"

#=============Text cleaning=============
def clean_text(text: str) -> str:
    """ Normalise a raw text field into a clean, model-ready string

    Lowercases, replaces URLs and email addresses with [URL]/[EMAIL] tokens
    to reduce domain memorisation, strips punctuation, and collapses whitespace
    """
    
    # safely handle NaN or non-string values
    text = "" if pd.isna(text) else str(text)

    # lowercase for vocabulary consistency
    text = text.lower()

    # anonymise / normalise URLs and email addresses with placeholders
    # this prevents the model from memorising specific domains or addresses and reduces leakage between the training and test splits
    text = re.sub(r"(https?://\S+|www\.\S+)", " [URL] ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " [EMAIL] ", text)

    # remove anything that isnt a letter, digit, whitespace, or square bracket
    # keep square brackets so [URL]/[EMAIL] tokens survive
    text = re.sub(r"[^a-z0-9\s\[\]]+", " ", text)

    #Collapse multiple consecutive spaces / newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


#================Main pipeline==============
def main():
    df = pd.read_csv(INPUT_FILE)

    # standardise source values to lowercase with no surrounding whitespace, then map any short aliases to their names
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].replace({"ai": "ai_generated"})

    # fill NaN with empty string so clean_text and string operations never receive a float NaN value
    for col in ["sender", "subject", "email_text"]:
        df[col] = df[col].fillna("").astype(str)

    # produce a separately cleaned column for each text field so that experiments can compare body-only vs subject+body vs full combined input
    df["text_body"]    = df["email_text"].apply(clean_text)
    df["text_subject"] = df["subject"].apply(clean_text)
    df["text_sender"]  = df["sender"].apply(clean_text)

    # 'sender: / subject: / body:" prefixes give the model positional context
    df["text_combined"] = ("sender: "  + df["text_sender"]  + " subject: " + df["text_subject"] + " body: "    + df["text_body"])

    # descriptive statistics columns
    # computed on the raw/cleaned text so you can report corpus statistics (e.g. mean email length) and investigate length as a feature later
    df["body_length_chars"] = df["email_text"].fillna("").astype(str).apply(len)
    df["body_length_words"] = df["text_body"].apply(lambda x: len(x.split()) if x else 0)

    # save the output
    df.to_csv(OUTPUT_FILE, index=False)

    # print a quick summary
    print("Preprocessing complete!")
    print(f"Saved: {OUTPUT_FILE}")
    print("Example cleaned text (first 1 row):")
    print(df.loc[0, ["text_sender", "text_subject", "text_body"]].to_dict())
    print("\nBody length (words) summary:")
    print(df["body_length_words"].describe())


if __name__ == "__main__":
    main()
