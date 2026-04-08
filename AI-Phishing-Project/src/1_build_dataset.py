import pandas as pd
from pathlib import Path

# =====================Path configuration=============================
# resolve the project root by going two levels up from this files location
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# input directory containing the raw CSV files
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# output directory for the processed/merged dataset
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# create the processed directory if it doesn't already exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ======================File paths=================================
# the three source CSV files to be merged into one single dataset
INPUT_FILES = [RAW_DIR / "ai_generated.csv", RAW_DIR / "real_phishing.csv", RAW_DIR / "legitimate.csv",]

# this is the final merged output file
OUTPUT_FILE = PROCESSED_DIR / "dataset_final.csv"

# ============================Schema definition===========================
# all columns that must be present in each input CSV; any extras are dropped
REQUIRED_COLUMNS = ["id", "sender", "subject", "email_text", "label", "source", "model", "prompt_type", "context", "date_collected",]

STRING_COLS = ["sender", "subject", "email_text", "source", "model", "prompt_type", "context", "date_collected",]


# =======================Helper functions========================
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV and validate it has the expected columns

    Raises FileNotFoundError if the file is missing
    ValueError if any required columns are absent
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    # check for any required columns that are absent in this file
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    # keep only the required columns
    df = df[REQUIRED_COLUMNS].copy()
    return df


# ==================Main pipeline=========================
def main():
    # load and concatenate all three source files into one DataFrame
    dfs = [load_csv(p) for p in INPUT_FILES]
    combined = pd.concat(dfs, ignore_index=True)

    # string column cleanup
    # strip whitespace and normalise nulls across all text fields
    for c in ["sender", "subject", "email_text", "source", "model", "prompt_type", "context", "date_collected"]:
        combined[c] = combined[c].fillna("").astype(str).str.strip()

    # coerce labels: anything non-numeric becomes -1 so it fails the range check below
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(-1).astype(int)

    # reject the dataset if any label falls outside the expected 0 (legitimate) / 1 (phishing) range
    bad = combined[~combined["label"].isin([0, 1])]
    if not bad.empty:
        raise ValueError(f"Invalid label values found: {bad['label'].unique().tolist()}")

    # ID validation
    # coerce IDs and check they're all numeric integers
    combined["id"] = pd.to_numeric(combined["id"], errors="coerce")
    if combined["id"].isna().any():
        raise ValueError("Some id values are not numeric. Please fix ids in your CSVs.")
    combined["id"] = combined["id"].astype(int)

    # make sure every ID is unique across the merged dataset
    dupes = combined[combined.duplicated(subset=["id"], keep=False)]
    if not dupes.empty:
        example = dupes["id"].head(10).tolist()
        raise ValueError(f"Duplicate ids detected. Example: {example}")

    # drop rows with empty bodies as they have no training signal
    before = len(combined)
    combined = combined[combined["email_text"].str.len() > 0].copy()
    removed_empty = before - len(combined)

    # save output
    combined.to_csv(OUTPUT_FILE, index=False)

    print("dataset_final.csv has been created.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(combined)} (removed empty bodies: {removed_empty})")
    print("Class balance (label counts):")
    print(combined["label"].value_counts())
    print("\nSource balance:")
    print(combined["source"].value_counts())


if __name__ == "__main__":
    main()
