"""Stage 2: Clean text and split into train/test."""

import csv
import os
import re

import numpy as np
import yaml


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data"]

    rng = np.random.default_rng(params["random_seed"])

    with open("data/raw/reviews.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    indices = np.arange(len(rows))
    rng.shuffle(indices)

    split_idx = int(len(rows) * (1 - params["test_split"]))

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/corpus.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "split"])

        for i, idx in enumerate(indices):
            row = rows[idx]
            split = "train" if i < split_idx else "test"
            writer.writerow([clean_text(row["text"]), row["label"], split])

    print(f"Prepared {split_idx} train / {len(rows) - split_idx} test -> data/processed/corpus.csv")


if __name__ == "__main__":
    main()
