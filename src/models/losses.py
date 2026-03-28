"""Loss functions for binary classification."""

import numpy as np


def binary_cross_entropy(y_pred, y_true):
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def binary_cross_entropy_grad(y_pred, y_true):
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return (-y_true / p + (1 - y_true) / (1 - p)) / y_true.shape[0]
