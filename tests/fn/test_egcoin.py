"""Tests for egcoin."""

import numpy as np
import pytest

from morie.fn.egcoin import engle_granger_2step

def test_egcoin_basic():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(300))
    y = 2.0 * x + rng.standard_normal(300)
    out = engle_granger_2step(y, x)
    assert out["cointegrated_5pct"] is True
    assert out["beta"][0] == pytest.approx(2.0, abs=0.15)


def test_egcoin_edge():
    # independent random walks must NOT test as cointegrated
    rng = np.random.default_rng(7)
    a = np.cumsum(rng.standard_normal(300))
    b = np.cumsum(rng.standard_normal(300))
    assert engle_granger_2step(a, b)["cointegrated_5pct"] is False
    with pytest.raises(ValueError):
        engle_granger_2step(np.ones(5), np.ones(5))
