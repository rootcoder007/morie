"""Tests for morie.fn.slxvar."""

from morie.fn import _array_core as np
from morie.fn.slxvar import slxvar


class TestSlxvar:
    def test_basic(self):
        np.random.seed(43)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        W = np.array([[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)])
        W = W / W.sum(axis=1, keepdims=True)
        sigma2 = 1.0
        result = slxvar(X, W, sigma2)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(43)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        W = np.array([[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)])
        W = W / W.sum(axis=1, keepdims=True)
        sigma2 = 1.0
        result = slxvar(X, W, sigma2)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(43)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        W = np.array([[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)])
        W = W / W.sum(axis=1, keepdims=True)
        sigma2 = 1.0
        result = slxvar(X, W, sigma2)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
