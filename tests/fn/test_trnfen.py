"""Tests for trnfen.transfer_entropy."""

from morie.fn import _array_core as np
import pytest

from morie.fn.granci import granger_causality_info
from morie.fn.trnfen import transfer_entropy


def test_trnfen_basic():
    rng = np.random.default_rng(42)
    n = 2000
    x = np.zeros(n); y = np.zeros(n)
    ex = rng.normal(size=n); ey = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.4 * y[t - 1] + 0.7 * x[t - 1] + ey[t]
    te = transfer_entropy(x, y, method="gaussian")
    assert te["te"] == pytest.approx(granger_causality_info(x, y, lag=1)["mi"])
    assert te["p_value"] < 0.01
    binned = transfer_entropy(x, y, method="binned", bins=4)
    assert binned["te"] > transfer_entropy(y, x, method="binned", bins=4)["te"]


def test_trnfen_edge():
    with pytest.raises(ValueError):
        transfer_entropy([1.0] * 100, [1.0] * 100, method="wavelet")
    with pytest.raises(ValueError):
        transfer_entropy([1.0] * 30, [1.0] * 30, method="binned", bins=4)  # too short
