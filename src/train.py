"""Stage 4: Train the feedforward neural network."""

import numpy as np
import yaml

from .models.losses import binary_cross_entropy, binary_cross_entropy_grad
from .models.network import FeedForwardNetwork
from .models.optimizers import SGD


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    X_train = np.load("data/features/X_train.npy")
    y_train = np.load("data/features/y_train.npy")

    input_dim = X_train.shape[1]
    layer_sizes = [input_dim] + params["hidden_layers"] + [1]

    net = FeedForwardNetwork(
        layer_sizes, activation=params["activation"], seed=params["random_seed"]
    )
    optimizer = SGD(learning_rate=params["learning_rate"], momentum=params["momentum"])

    rng = np.random.default_rng(params["random_seed"])
    n = X_train.shape[0]
    batch_size = params["batch_size"]

    for epoch in range(1, params["epochs"] + 1):
        indices = np.arange(n)
        rng.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            y_pred = net.forward(X_batch)
            loss = binary_cross_entropy(y_pred, y_batch)
            grad = binary_cross_entropy_grad(y_pred, y_batch)

            net.backward(grad)
            optimizer.step(net.layers)

            epoch_loss += loss
            n_batches += 1

        print(f"Epoch {epoch}/{params['epochs']}  loss={epoch_loss / n_batches:.4f}")

    net.save("models/model.npz")
    print("Model saved -> models/model.npz")


if __name__ == "__main__":
    main()
