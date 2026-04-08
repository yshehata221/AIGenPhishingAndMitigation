import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# ===========Path configuration=============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_preprocessed.csv"


# ============Load the data===========
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
print("Dataset size:", len(df))


# ===========Define features and labels=====================
# use the combined text field (sender + subject + body) as the model input
# sender + subject + body, built in preprocessing
X = df["text_combined"]

# 0 = legit & 1 = phishing
y = df["label"]


# ====================TF-IDF vectorisation===========
# TF-IDF (Term Frequency–Inverse Document Frequency) converts raw text into a numeric matrix where each column represents a vocabulary term and each value reflects how distinctive that term is for a given email
# max_features=5000 - caps the vocabulary at the 5000 highest scoring terms, keeping the feature matrix manageable and reducing noise from rare words
# stop_words="english" - removes common English words (e.g "the", "and") that carry no signal for phishing detection.
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

print("Creating TF-IDF features...")

# fit_transform learns the vocabulary from X and immediately transforms it into the TF-IDF matrix. Returns a sparse matrix of shape (n_emails, n_features).
X_tfidf = vectorizer.fit_transform(X)
print("TF-IDF shape:", X_tfidf.shape)

# ==============Vocabulary inspection================
# check that vocab contains real words, not punctuation artifacts
feature_names = vectorizer.get_feature_names_out()
print("\\nExample vocabulary terms:")
print(feature_names[:20])
# print the first 20 terms in alphabetical order
