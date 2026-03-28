"""SGD optimizer with optional momentum."""

import numpy as np

from .layers import DenseLayer


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocities = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if not isinstance(layer, DenseLayer):
                continue

            key_w = (i, "W")
            key_b = (i, "b")

            if key_w not in self._velocities:
                self._velocities[key_w] = np.zeros_like(layer.W)
                self._velocities[key_b] = np.zeros_like(layer.b)

            self._velocities[key_w] = (
                self.momentum * self._velocities[key_w] - self.lr * layer.dW
            )
            self._velocities[key_b] = (
                self.momentum * self._velocities[key_b] - self.lr * layer.db
            )

            layer.W += self._velocities[key_w]
            layer.b += self._velocities[key_b]
