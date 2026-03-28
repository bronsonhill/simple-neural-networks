"""Stage 5: Evaluate model and produce metrics.json."""

import json
import os

import numpy as np
import yaml

from .models.losses import binary_cross_entropy
from .models.network import FeedForwardNetwork


def main():
    with open("params.yaml") as f:
        all_params = yaml.safe_load(f)

    train_params = all_params["train"]
    threshold = all_params["evaluate"]["threshold"]

    X_test = np.load("data/features/X_test.npy")
    y_test = np.load("data/features/y_test.npy")

    input_dim = X_test.shape[1]
    layer_sizes = [input_dim] + train_params["hidden_layers"] + [1]

    net = FeedForwardNetwork(
        layer_sizes,
        activation=train_params["activation"],
        seed=train_params["random_seed"],
    )
    net.load("models/model.npz")

    y_pred = net.forward(X_test)
    y_class = (y_pred >= threshold).astype(np.float64)

    tp = ((y_class == 1) & (y_test == 1)).sum()
    fp = ((y_class == 1) & (y_test == 0)).sum()
    fn = ((y_class == 0) & (y_test == 1)).sum()

    accuracy = (y_class == y_test).mean()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    test_loss = binary_cross_entropy(y_pred, y_test)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "test_loss": round(float(test_loss), 4),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
