"""Feedforward neural network assembled from layers."""

import numpy as np

from .layers import DenseLayer, ReLU, Sigmoid


ACTIVATIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
}


class FeedForwardNetwork:
    def __init__(self, layer_sizes, activation="relu", seed=42):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.layers = []

        act_cls = ACTIVATIONS[activation]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(layer_sizes[i], layer_sizes[i + 1], seed=seed + i)
            )
            if i < len(layer_sizes) - 2:
                self.layers.append(act_cls())
            else:
                self.layers.append(Sigmoid())

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                params[f"W_{i}"] = layer.W
                params[f"b_{i}"] = layer.b
        np.savez(path, **params)

    def load(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                layer.W = data[f"W_{i}"]
                layer.b = data[f"b_{i}"]
