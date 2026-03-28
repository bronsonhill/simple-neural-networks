"""Stage 1: Generate a synthetic sentiment dataset."""

import csv
import os

import numpy as np
import yaml


POSITIVE_PHRASES = [
    "great movie", "loved it", "excellent film", "wonderful story",
    "amazing acting", "highly recommend", "beautiful cinematography",
    "perfect ending", "brilliant performance", "very enjoyable",
    "fantastic plot", "superb direction", "heartwarming tale",
    "outstanding cast", "must watch", "truly inspiring",
]

NEGATIVE_PHRASES = [
    "terrible movie", "waste of time", "boring film", "awful story",
    "bad acting", "do not recommend", "poor cinematography",
    "horrible ending", "weak performance", "very disappointing",
    "predictable plot", "terrible direction", "depressing mess",
    "awful cast", "avoid this", "truly awful",
]

FILLER_WORDS = [
    "the", "a", "this", "that", "really", "very", "so", "quite",
    "absolutely", "just", "was", "is", "it", "i", "we", "they",
    "think", "felt", "movie", "film", "story", "watched", "saw",
]


def generate_review(rng, phrases, min_words=8, max_words=25):
    num_phrases = rng.integers(1, 4)
    chosen = rng.choice(len(phrases), size=num_phrases, replace=True)
    parts = [phrases[i] for i in chosen]

    target_len = rng.integers(min_words, max_words + 1)
    while len(" ".join(parts).split()) < target_len:
        parts.insert(
            rng.integers(0, len(parts) + 1),
            FILLER_WORDS[rng.integers(0, len(FILLER_WORDS))],
        )

    return " ".join(parts)


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data"]

    rng = np.random.default_rng(params["random_seed"])
    num_samples = params["num_samples"]

    os.makedirs("data/raw", exist_ok=True)

    with open("data/raw/reviews.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for _ in range(num_samples // 2):
            writer.writerow([generate_review(rng, POSITIVE_PHRASES), 1])
            writer.writerow([generate_review(rng, NEGATIVE_PHRASES), 0])

    print(f"Generated {num_samples} reviews -> data/raw/reviews.csv")


if __name__ == "__main__":
    main()
