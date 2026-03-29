"""Stage 3: Convert text to numerical feature vectors (BoW, TF-IDF, or embeddings)."""

import csv
import io
import json
import os
import zipfile
from collections import Counter
from urllib.request import urlretrieve

import numpy as np
import yaml

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


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


def load_glove(dim):
    """Download and load GloVe embeddings, caching to data/embeddings/."""
    cache_dir = "data/embeddings"
    npy_path = os.path.join(cache_dir, f"glove.6B.{dim}d.npy")
    vocab_path = os.path.join(cache_dir, f"glove.6B.{dim}d.vocab.json")

    if os.path.exists(npy_path) and os.path.exists(vocab_path):
        print(f"Loading cached GloVe {dim}d embeddings...")
        vectors = np.load(npy_path)
        with open(vocab_path) as f:
            vocab = json.load(f)
        return vocab, vectors

    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "glove.6B.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading GloVe 6B embeddings (~862MB)...")
        urlretrieve(GLOVE_URL, zip_path)

    target = f"glove.6B.{dim}d.txt"
    print(f"Extracting {target}...")

    vocab = {}
    vectors = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(target) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                parts = line.strip().split()
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float64)
                vocab[word] = len(vocab)
                vectors.append(vec)

    vectors = np.array(vectors)
    np.save(npy_path, vectors)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    print(f"Loaded {len(vocab)} GloVe vectors ({dim}d)")
    return vocab, vectors


def texts_to_embeddings(texts, glove_vocab, glove_vectors, pooling="mean"):
    """Represent each text by pooling its word embeddings.

    pooling: "mean", "max", or "sum"
    """
    dim = glove_vectors.shape[1]
    X = np.zeros((len(texts), dim), dtype=np.float64)

    for i, text in enumerate(texts):
        words = text.split()
        vecs = [glove_vectors[glove_vocab[w]] for w in words if w in glove_vocab]
        if vecs:
            stacked = np.array(vecs)
            if pooling == "mean":
                X[i] = stacked.mean(axis=0)
            elif pooling == "max":
                X[i] = stacked.max(axis=0)
            elif pooling == "sum":
                X[i] = stacked.sum(axis=0)

    return X


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

    method = params["method"]

    if method == "embedding":
        dim = params.get("embedding_dim", 50)
        pooling = params.get("embedding_pooling", "mean")
        glove_vocab, glove_vectors = load_glove(dim)
        X_train = texts_to_embeddings(train_texts, glove_vocab, glove_vectors, pooling)
        X_test = texts_to_embeddings(test_texts, glove_vocab, glove_vectors, pooling)
        vocab = {w: i for i, w in enumerate(list(glove_vocab.keys())[:params["max_vocab_size"]])}
    else:
        vocab = build_vocabulary(
            train_texts, params["max_vocab_size"], params["min_doc_freq"]
        )
        X_train = texts_to_bow(train_texts, vocab)
        X_test = texts_to_bow(test_texts, vocab)

        if method == "tfidf":
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
        f"method={method}"
    )


if __name__ == "__main__":
    main()
