import re
import pandas as pd
from pathlib import Path

#---------------------------------------------------------------------------
#Path configuration
#---------------------------------------------------------------------------

#Resolve the project root two levels up from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

#Read from the merged dataset produced by the previous pipeline step
INPUT_FILE = PROCESSED_DIR / "dataset_final.csv"

#Write the cleaned, feature-enriched dataset here
OUTPUT_FILE = PROCESSED_DIR / "dataset_preprocessed.csv"


#---------------------------------------------------------------------------
#Text cleaning
#---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
Normalise a raw text field into a clean, model-ready string.

Steps applied (in order):
    1. Coerce non-string / NaN values to an empty string.
    2. Lowercase everything for vocabulary consistency.
    3. Replace URLs and email addresses with placeholder tokens
        ([URL], [EMAIL]) — reduces data leakage and improves generalisation.
    4. Strip punctuation and special characters, keeping only
        alphanumerics, whitespace, and square brackets (used by tokens).
    5. Collapse runs of whitespace into a single space.

Args:
    text: Raw string from a DataFrame cell.

Returns:
    Cleaned, lowercase string.
    """
    
    #Safely handle NaN or non-string values
    text = "" if pd.isna(text) else str(text)

    #Lowercase for vocabulary consistency
    text = text.lower()

    #Anonymise / normalise URLs and email addresses with placeholder tokens.
    #This prevents the model from memorising specific domains or addresses and reduces leakage between the training and test splits.
    text = re.sub(r"(https?://\S+|www\.\S+)", " [URL] ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " [EMAIL] ", text)

    #Remove anything that isn't a letter, digit, whitespace, or square bracket.
    #Square brackets are kept so the [URL] / [EMAIL] tokens survive intact.
    text = re.sub(r"[^a-z0-9\s\[\]]+", " ", text)

    #Collapse multiple consecutive spaces / newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


#---------------------------------------------------------------------------
#Main pipeline
#---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)

    #Source label normalisation
    #Standardise source values to lowercase with no surrounding whitespace, then map any short-form aliases to their canonical names.
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].replace({"ai": "ai_generated"})

    #Null-safety for text columns
    #Fill NaN with empty string so clean_text and string operations never receive a float NaN value.
    for col in ["sender", "subject", "email_text"]:
        df[col] = df[col].fillna("").astype(str)

    #Cleaned text fields
    #Produce a separately cleaned column for each text field so that experiments can compare body-only vs subject+body vs full combined input.
    df["text_body"]    = df["email_text"].apply(clean_text)
    df["text_subject"] = df["subject"].apply(clean_text)
    df["text_sender"]  = df["sender"].apply(clean_text)

    #Combined field that prefixes each section with a short label.
    #The prefix tokens ("sender:", "subject:", "body:") give the model positional context about which section it's reading.
    df["text_combined"] = (
        "sender: "  + df["text_sender"]  +
        " subject: " + df["text_subject"] +
        " body: "    + df["text_body"]
    )

    #Descriptive statistics columns
    #Computed on the raw / cleaned text so you can report corpus statistics (e.g. mean email length) and investigate length as a feature later.
    df["body_length_chars"] = df["email_text"].fillna("").astype(str).apply(len)
    df["body_length_words"] = df["text_body"].apply(lambda x: len(x.split()) if x else 0)

    #Save output
    df.to_csv(OUTPUT_FILE, index=False)

    #Print a quick sanity-check summary
    print("Preprocessing complete!!!")
    print(f"Saved: {OUTPUT_FILE}")
    print("Example cleaned text (first 1 row):")
    print(df.loc[0, ["text_sender", "text_subject", "text_body"]].to_dict())
    print("\nBody length (words) summary:")
    print(df["body_length_words"].describe())


if __name__ == "__main__":
    main()
