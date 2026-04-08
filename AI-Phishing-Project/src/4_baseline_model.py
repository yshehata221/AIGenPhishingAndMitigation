import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ==============Path config==============
# project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# load the preprocessed dataset produced by the previous pipeline step
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_preprocessed.csv"


# ==============Load data==============
print("Loading the dataset...")
df = pd.read_csv(DATA_FILE)

# sender + subject + body concatenated
X = df["text_combined"]

# 1 = phishing & 0 = legit
y = df["label"]


# ==============TF-IDF vectorisation==============
# convert the raw text into a matrix of TF-IDF features
# cap at 5000 terms to limit memory usage and drop stopwords
# stop_words="english" removes common English words (e.g. "the", "and") that arent valuable
print("Vectorising the text...")
vectoriser = TfidfVectorizer(max_features=5000, stop_words="english")

# fit_transform learns the vocabulary from X and transforms it into the TF-IDF matrix in a single pass then returns a matrix of (n_emails × n_features)
X_tfidf = vectoriser.fit_transform(X)


# ==============Train/test split==============
# 80/20 split amd fixed seed so results are reproducible
# random_state=42 ensures the split is reproducible across runs
print("Splitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


# ==============Model training==============
# logistic Regression is used as an interpretable baseline classifier.
# its fast to train, produces calibrated probabilities, and its coefficients can be inspected to see which terms most strongly predict phishing
# max_iter=1000 avoids ConvergenceWarning on larger vocab sizes.
print("Training baseline classifier...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained.")


# ==============Evaluation==============
print("\\nEvaluating model......")

# generate predictions on the test set
y_pred = model.predict(X_test)

# classification report shows per-class precision, recall, and F1-score.
# recall on class 1 matters more, a missed phishing email costs more than a false alarm
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# the confusion matrix breaks down predictions into true positives, true negatives, false positives, and false negatives
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
