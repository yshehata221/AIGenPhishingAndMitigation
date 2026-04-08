import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ==========Path config==============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_preprocessed.csv"

df = pd.read_csv(DATA_FILE)

# ensure text fields contain valid strings (replace NaN with empty strings)
df["text_body"] = df["text_body"].fillna("").astype(str)
df["text_subject"] = df["text_subject"].fillna("").astype(str)
df["text_combined"] = df["text_combined"].fillna("").astype(str)


# ============Experiment definitions================
# each experiment varies the input fields and/or ngram range;
# fixed seed means all four are tested on identical emails
experiments = [
    ("Body Only", df["text_body"], (1, 1)),
    ("Subject + Body", df["text_subject"] + " " + df["text_body"], (1, 1)),
    ("Sender + Subject + Body", df["text_combined"], (1, 1)),
    ("Sender + Subject + Body + Bigrams", df["text_combined"], (1, 2)),
]

# 0 = legit & 1 = phishing
y = df["label"]


# ============Experiment loop=======================
# each experiment is run independently with its own vectoriser and model so that the vocabulary and weights are learned only from that input combination
for name, text_data, ngram_range in experiments:

    print("\n============================")
    print("Experiment:", name)
    print("============================")

    # Vectorise the text input for this experiment using TF-IDF.
    # max_features=5000 — keeps only the 5,000 most informative terms.
    # stop_words="english" — removes common words
    # ngram_range controls whether the model uses single words (1,1) or also includes word pairs (1,2)
    vectoriser = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=ngram_range)
    X = vectoriser.fit_transform(text_data)

    # hold out 20% of the data for evaluation
    # random_state=42 is fixed across all experiments so that each model is tested on the same emails
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression is used as an interpretable baseline classifier
    # max_iter=1000 gives the solver enough iterations to converge on a vocabulary of this size without raising a ConvergenceWarning
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # recall on class 1 matters most, a missed phishing costs more than a false alarm
    print(classification_report(y_test, y_pred))
