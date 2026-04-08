import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =============================Path config=================================
# resolve the project root just two levels up from this files location
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# load the preprocessed dataset that was produced by the previous pipeline step
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_preprocessed.csv"


# ======================Load and clean data==============
df = pd.read_csv(DATA_FILE)

# ensure the vectoriser never gets a non-string value
df["text_body"]     = df["text_body"].fillna("").astype(str)
df["text_combined"] = df["text_combined"].fillna("").astype(str)

# 0 = legit & 1 = phishing
y = df["label"]

#  ========================Ablation experiment definitions====================
# compare full input vs body-only to see how much sender + subject contribute
experiments = {"Full Features (Sender + Subject + Body)": df["text_combined"], "Ablation (Body Only)": df["text_body"],}

# =========================Experimental loop====================
# each experiment is run independently with its own vectoriser and model so that the vocabulary and weights are learned only from that specific input combination
for name, text_data in experiments.items():
    print("\n============================")
    print("Experiment:", name)
    print("============================")

    # vectorise the text input for this experiment using TF-IDF
    # max_features=5000 - keeps only the 5000 most informative terms
    # stop_words="english" - removes common words with no discriminative signal
    # ngram_range=(1,2) picks up two-word phishing phrases like "click here" or "verify account"
    vectoriser = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = vectoriser.fit_transform(text_data)

    # hold out 20% of the data for evaluation
    # same seed across both runs so they're tested on identical emails
    # random_state=42 is fixed across both experiments so each model is tested on the same emails
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train a logistic regression classifier as an interpretable baseline
    # max_iter=1000 gives the solver enough iterations to converge on a vocabulary of this size without raising a ConvergenceWarning
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # score gap between experiments = value added by sender + subject fields
    # the difference in scores between the two experiments reveals how much predictive value the sender and the subject fields add over just the body alone
    print(classification_report(y_test, y_pred))
