"""Stage 3: Convert text to numerical feature vectors (BoW or TF-IDF)."""

import csv
import json
import os
from collections import Counter

import numpy as np
import yaml


def build_vocabulary(texts, max_vocab_size, min_doc_freq):
    doc_freq = Counter()
    for text in texts:
        doc_freq.update(set(text.split()))

    filtered = {
        word: freq for word, freq in doc_freq.items() if freq >= min_doc_freq
    }
    sorted_words = sorted(filtered, key=lambda w: -filtered[w])[:max_vocab_size]
    return {word: idx for idx, word in enumerate(sorted_words)}


def texts_to_bow(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, text in enumerate(texts):
        for word in text.split():
            if word in vocab:
                X[i, vocab[word]] += 1
    return X


def bow_to_tfidf(X, eps=1e-7):
    tf = X / (X.sum(axis=1, keepdims=True) + eps)
    n_docs = X.shape[0]
    df = (X > 0).sum(axis=0)
    idf = np.log(n_docs / (df + eps))
    return tf * idf


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["features"]

    with open("data/processed/corpus.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    train_texts = [r["text"] for r in rows if r["split"] == "train"]
    train_labels = [int(r["label"]) for r in rows if r["split"] == "train"]
    test_texts = [r["text"] for r in rows if r["split"] == "test"]
    test_labels = [int(r["label"]) for r in rows if r["split"] == "test"]

    vocab = build_vocabulary(
        train_texts, params["max_vocab_size"], params["min_doc_freq"]
    )

    X_train = texts_to_bow(train_texts, vocab)
    X_test = texts_to_bow(test_texts, vocab)

    if params["method"] == "tfidf":
        X_train = bow_to_tfidf(X_train)
        X_test = bow_to_tfidf(X_test)

    y_train = np.array(train_labels, dtype=np.float64).reshape(-1, 1)
    y_test = np.array(test_labels, dtype=np.float64).reshape(-1, 1)

    os.makedirs("data/features", exist_ok=True)
    np.save("data/features/X_train.npy", X_train)
    np.save("data/features/X_test.npy", X_test)
    np.save("data/features/y_train.npy", y_train)
    np.save("data/features/y_test.npy", y_test)

    with open("data/features/vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print(
        f"Vectorized: X_train {X_train.shape}, X_test {X_test.shape}, "
        f"vocab size {len(vocab)}"
    )


if __name__ == "__main__":
    main()
