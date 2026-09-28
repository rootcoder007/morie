"""Tests for morie.fn.semrlm."""

from morie.fn import _array_core as np
from morie.fn.semrlm import semrlm


class TestSemrlm:
    def test_basic(self):
        np.random.seed(12)
        n = 25
        resid = np.random.randn(n)
        W = np.array([[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)])
        W = W / W.sum(axis=1, keepdims=True)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        result = semrlm(resid, X, W)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(12)
        n = 25
        resid = np.random.randn(n)
        W = np.array([[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)])
        W = W / W.sum(axis=1, keepdims=True)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        result = semrlm(resid, X, W)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(12)
        n = 25
        resid = np.random.randn(n)
        W = np.array([[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)])
        W = W / W.sum(axis=1, keepdims=True)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        result = semrlm(resid, X, W)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
