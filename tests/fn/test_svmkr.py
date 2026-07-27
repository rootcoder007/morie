"""Tests for svmkr.svm_kernel_trick."""

import numpy as np
import pytest

from morie.fn.svmkr import svm_kernel_trick


def _rings(seed=0, n=240):
    """Two concentric rings -- not linearly separable by construction."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    radius = np.where(np.arange(n) % 2 == 0, 1.0, 3.0)
    X = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    X += rng.normal(0, 0.15, X.shape)
    return X, (np.arange(n) % 2).astype(int)


def test_svmkr_rbf_solves_what_linear_cannot():
    """The kernel is the whole point: on concentric rings the RBF machine
    is near-perfect while the linear one hovers at chance. If the kernel
    argument were ignored, these two numbers would coincide."""
    X, y = _rings()
    rbf = svm_kernel_trick(X, y, kernel="rbf", C=1.0)
    lin = svm_kernel_trick(X, y, kernel="linear", C=1.0)
    assert float(rbf["train_accuracy"]) >= 0.97
    # Measured: linear reaches 0.68 on this fixture (angle imbalance gives
    # it a bit over chance); the discriminating fact is the GAP.
    assert float(lin["train_accuracy"]) <= 0.75
    assert float(rbf["train_accuracy"]) - float(lin["train_accuracy"]) > 0.25


def test_svmkr_polynomial_kernel_also_separates_the_rings():
    """A degree-2 polynomial kernel contains x^2 + y^2, which is exactly
    the ring radius, so it must also succeed."""
    X, y = _rings(seed=1)
    poly = svm_kernel_trick(X, y, kernel="poly", degree=2, C=1.0)
    assert float(poly["train_accuracy"]) >= 0.95


def test_svmkr_support_vectors_are_a_subset_of_the_data():
    X, y = _rings(seed=2)
    r = svm_kernel_trick(X, y, kernel="rbf")
    assert sum(r["n_support"]) <= X.shape[0]
    assert all(c >= 1 for c in r["n_support"])
