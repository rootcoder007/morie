"""Tests for copgmb."""

import numpy as np
import pytest

from morie.fn.copgmb import gumbel_copula

def test_copgmb_basic():
    out = gumbel_copula(0.5, 0.5, 4.0)
    assert out["tau"] == pytest.approx(0.75)  # 1 - 1/theta
    assert out["cdf"] < 0.5


def test_copgmb_edge():
    assert gumbel_copula(0.4, 0.6, 1.0)["cdf"] == pytest.approx(0.24, abs=1e-8)
    with pytest.raises(ValueError):
        gumbel_copula(0.5, 0.5, 0.5)
