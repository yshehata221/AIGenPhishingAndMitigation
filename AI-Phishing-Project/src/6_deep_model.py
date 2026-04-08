import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =================Path configuration=====================
# resolve the project root two levels up from this filess location
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# load the preprocessed dataset produced by the previous pipeline step
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_preprocessed.csv"

# ================Load data=======================
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# text_combined was the best-performing input in the TF-IDF baseline experiments
# fill any NaN values to avoid errors during tokenisation.
df["text_combined"] = df["text_combined"].fillna("").astype(str)
X = df["text_combined"]

# 0 = legit & 1 = phishing
y = df["label"]

# ======================Train/test split==================
# hold out 20% of the data for final evaluation
# random_state=42 keeps the split consistent with the baseline experiments
# fixed seed so results are comparable with the baseline models
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =============Tokenisation==================
# vocab size: words beyond this rank are discarded
max_words = 10000

# sequences are padded or shortened to this length
max_len = 200

# fit tokeniser on training corpus only; <OOV> catches unseen words at inference
print("Tokenising text...")
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# convert each email from a string into a list of integer token indices
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

# post-padding/truncation preserves the start of each email
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad  = pad_sequences(X_test_seq,  maxlen=max_len, padding="post", truncating="post")

# ==================Model architecture===================
# BiLSTM reads sequences in both directions
# useful for catching suspicious phrases that appear late in a long email body
print("Building BiLSTM model...")
model = Sequential([
    # embedding layer learns a dense 64 dimensional vector for each token
    # input_dim=max_words limits the vocabulary to the same size used by the tokeniser
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),

    # bidirectional LSTM processes the sequence forwards and backwards, concatenating both hidden states into a single output vector
    Bidirectional(LSTM(64, return_sequences=False)),
    # return_sequences=False returns only the final hidden state

    # dropout randomly zeros 30% of activations during training to reduce overfitting on the set
    Dropout(0.3),

    # fully connected layer to learn a non-linear combination of LSTM features
    Dense(32, activation="relu"),

    # single sigmoid output produces a probability in 0 & 1 for binary classification
    # probability > 0.5 is phishing
    Dense(1, activation="sigmoid")
])

# binary_crossentropy is the standard loss function for binary classification
# adam is an adaptive learning rate optimiser well suited to sequence models
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# ====================Training===================
# train for 3 epochs
# batch_size=32 is a standard default that balances stability and speed
# validation_split=0.2 reserves 20% of the training data to monitor
# validation loss and accuracy after each epoch
print("Training model...")
history = model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)

# =================Evaluation=====================
print("Evaluating model...")

# predict() returns a probability for each email
threshold at 0.5 to convert to binary class predictions.
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

# recall on class 1 matters more as missed phishing costs more than a false alarm
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# breakdown of true positives, true negatives, false positives, and false negatives
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
