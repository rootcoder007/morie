"""Tests for wnoml."""

import numpy as np
import pytest

from morie.fn.wnoml import wnominate_logit


def test_wnoml_basic():
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    Z = np.array([[[-1.0], [1.0]]])
    V = np.array([[1.0], [1.0], [0.0], [0.0]])
    out = wnominate_logit(V, X, Z, beta=50.0)
    assert out["correct_classification"] == pytest.approx(1.0)
    assert out["loglik"] < 0


def test_wnoml_edge():
    X = np.array([[-1.0], [1.0]])
    Z = np.array([[[-1.0], [1.0]]])
    with pytest.raises(ValueError):
        wnominate_logit(np.full((2, 1), np.nan), X, Z)  # all missing
    with pytest.raises(ValueError):
        wnominate_logit(np.ones((2, 1)), X, np.zeros((1, 3, 1)))  # bad Z shape
