import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_preprocessed.csv"

df = pd.read_csv(DATA_FILE)

df["text_combined"] = df["text_combined"].fillna("").astype(str)
df["model"]  = df["model"].fillna("none").astype(str).str.strip().str.lower()
df["source"] = df["source"].fillna("").astype(str).str.strip().str.lower()

phishing_df = df[df["label"] == 1]
legit_df    = df[df["label"] == 0]

# vocab cap; rarer words are discarded
max_words = 10000
# sequences are padded or truncated to this length
max_len   = 200 

# each iteration holds out one generator entirely from training and tests on its emails only, simulating detection of a phishing source
for TEST_MODEL in ["deepseek", "claude", "gemini", "grok", "gpt"]:
    print(f"\n{'='*50}")
    print(f"Held-out model: {TEST_MODEL.upper()}")
    print(f"{'='*50}")

    train_phishing = phishing_df[phishing_df["model"] != TEST_MODEL]
    test_phishing  = phishing_df[phishing_df["model"] == TEST_MODEL]

    if len(test_phishing) == 0:
        print(f"No test emails found for {TEST_MODEL}, skipping.")
        continue

    # match legitimate test size to phishing to keep the test set balanced 50/50
    test_legit  = legit_df.sample(n=len(test_phishing), random_state=42)
    train_legit = legit_df.drop(test_legit.index)

    train_df = pd.concat([train_phishing, train_legit]).sample(frac=1, random_state=42)
    test_df  = pd.concat([test_phishing,  test_legit]).sample(frac=1, random_state=42)

    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    X_train, y_train = train_df["text_combined"], train_df["label"]
    X_test,  y_test  = test_df["text_combined"],  test_df["label"]

    # fit tokeniser on training corpus only so no test data leaks into the vocabulary
    tokeniser = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokeniser.fit_on_texts(X_train)

    X_train_pad = pad_sequences(tokeniser.texts_to_sequences(X_train), maxlen=max_len, padding="post")
    X_test_pad  = pad_sequences(tokeniser.texts_to_sequences(X_test),  maxlen=max_len, padding="post")

    # fresh model per iteration so weights from previous generators dont carry over
    model = Sequential([
        Embedding(max_words, 64),
        # reads sequence both ways to capture context
        Bidirectional(LSTM(64)),  
        Dropout(0.3),
        Dense(32, activation="relu"),
        # > 0.5 → phishing
        Dense(1, activation="sigmoid")  
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)

    # recall on class 1 matters most as missed phishing costs more than a false alarm
    y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
    print(f"\nClassification Report — held out: {TEST_MODEL.upper()}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
