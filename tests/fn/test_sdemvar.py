"""Tests for morie.fn.sdemvar."""

from morie.fn import _array_core as np
from morie.fn.sdemvar import sdemvar


class TestSdemvar:
    def test_basic(self):
        np.random.seed(49)
        n = 15
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        W = np.array(
            [[0.5 if abs(i - j) in (1, n - 1) else 0.0 for j in range(n)] for i in range(n)]
        )  # row-standardised ring
        WX = W @ X[:, 1:]  # lag of the non-constant column; X * 0.3 was collinear with X
        lam = 0.3
        sigma2 = 1.0
        result = sdemvar(X, WX, W, lam, sigma2)
        assert result is not None

    def test_returns_spatial_result(self):
        np.random.seed(49)
        n = 15
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        W = np.array(
            [[0.5 if abs(i - j) in (1, n - 1) else 0.0 for j in range(n)] for i in range(n)]
        )  # row-standardised ring
        WX = W @ X[:, 1:]  # lag of the non-constant column; X * 0.3 was collinear with X
        lam = 0.3
        sigma2 = 1.0
        result = sdemvar(X, WX, W, lam, sigma2)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        np.random.seed(49)
        n = 15
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        W = np.array(
            [[0.5 if abs(i - j) in (1, n - 1) else 0.0 for j in range(n)] for i in range(n)]
        )  # row-standardised ring
        WX = W @ X[:, 1:]  # lag of the non-constant column; X * 0.3 was collinear with X
        lam = 0.3
        sigma2 = 1.0
        result = sdemvar(X, WX, W, lam, sigma2)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
