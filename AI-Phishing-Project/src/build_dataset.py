import pandas as pd
from pathlib import Path

#---------------------------------------------------------------------------
#Path configuration
#---------------------------------------------------------------------------

#Resolve the project root by going two levels up from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#Input directory containing the raw CSV files
RAW_DIR = PROJECT_ROOT / "data" / "raw"

#Output directory for the processed/merged dataset
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

#Create the processed directory if it doesn't already exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#---------------------------------------------------------------------------
#File paths
#---------------------------------------------------------------------------

#The three source CSV files to be merged into a single dataset
INPUT_FILES = [
    RAW_DIR / "ai_generated.csv",
    RAW_DIR / "real_phishing.csv",
    RAW_DIR / "legitimate.csv",
]

#Final merged output file
OUTPUT_FILE = PROCESSED_DIR / "dataset_final.csv"

#---------------------------------------------------------------------------
#Schema definition
#---------------------------------------------------------------------------

#All columns that must be present in each input CSV; any extras are dropped
REQUIRED_COLUMNS = [
    "id",
    "sender",
    "subject",
    "email_text",
    "label",
    "source",
    "model",
    "prompt_type",
    "context",
    "date_collected",
]


#---------------------------------------------------------------------------
#Helper functions
#---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load a single CSV file and validate its schema.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        A DataFrame containing only the required columns.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If any required columns are missing from the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    #Check for any required columns that are absent in this file
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    #Keep only the required columns (drops any extra columns in the source file)
    df = df[REQUIRED_COLUMNS].copy()
    return df


#---------------------------------------------------------------------------
#Main pipeline
#---------------------------------------------------------------------------

def main():
    #Load and concatenate all three source files into one DataFrame
    dfs = [load_csv(p) for p in INPUT_FILES]
    combined = pd.concat(dfs, ignore_index=True)

    #String column cleanup
    #Fill NaN values with empty string, cast to str, and strip leading/trailing whitespace
    for c in ["sender", "subject", "email_text", "source", "model", "prompt_type", "context", "date_collected"]:
        combined[c] = combined[c].fillna("").astype(str).str.strip()

    #Label validation
    #Coerce labels to numeric; anything that can't be converted becomes -1
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(-1).astype(int)

    #Reject the dataset if any label falls outside the expected 0 (legitimate) / 1 (phishing) range
    bad = combined[~combined["label"].isin([0, 1])]
    if not bad.empty:
        raise ValueError(f"Invalid label values found: {bad['label'].unique().tolist()}")

    #ID validation
    #Coerce IDs to numeric so we can detect non-integer values
    combined["id"] = pd.to_numeric(combined["id"], errors="coerce")
    if combined["id"].isna().any():
        raise ValueError("Some 'id' values are not numeric. Fix ids in your CSVs.")
    combined["id"] = combined["id"].astype(int)

    #Ensure every ID is unique across the merged dataset
    dupes = combined[combined.duplicated(subset=["id"], keep=False)]
    if not dupes.empty:
        example = dupes["id"].head(10).tolist()
        raise ValueError(f"Duplicate ids detected. Example: {example}")

    #Empty body removal
    #Drop rows where the email body is empty after stripping (they add no training signal)
    before = len(combined)
    combined = combined[combined["email_text"].str.len() > 0].copy()
    removed_empty = before - len(combined)

    #Save output
    combined.to_csv(OUTPUT_FILE, index=False)

    #Print a summary of the final dataset
    print("dataset_final.csv has created!!!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(combined)} (removed empty bodies: {removed_empty})")
    print("Class balance (label counts):")
    print(combined["label"].value_counts())
    print("\nSource balance:")
    print(combined["source"].value_counts())


if __name__ == "__main__":
    main()
