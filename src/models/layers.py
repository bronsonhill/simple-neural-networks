"""Dense layer and activation functions built from scratch in NumPy."""

import numpy as np


class DenseLayer:
    def __init__(self, input_dim, output_dim, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((input_dim, output_dim)) * np.sqrt(
            2.0 / input_dim
        )
        self.b = np.zeros((1, output_dim))
        self.dW = None
        self.db = None
        self._input = None

    def forward(self, X):
        self._input = X
        return X @ self.W + self.b

    def backward(self, dZ):
        m = dZ.shape[0]
        self.dW = self._input.T @ dZ / m
        self.db = dZ.sum(axis=0, keepdims=True) / m
        return dZ @ self.W.T


class ReLU:
    def __init__(self):
        self._input = None

    def forward(self, X):
        self._input = X
        return np.maximum(0, X)

    def backward(self, dout):
        return dout * (self._input > 0)


class Sigmoid:
    def __init__(self):
        self._output = None

    def forward(self, X):
        self._output = 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        return self._output

    def backward(self, dout):
        return dout * self._output * (1.0 - self._output)
