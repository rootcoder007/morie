"""Tests for blncop."""

from morie.fn import _array_core as np
import pytest

from morie.fn.blncop import blomqvists_beta_copula

def test_blncop_basic():
    assert blomqvists_beta_copula("independence")["beta"] == pytest.approx(0.0)
    assert blomqvists_beta_copula("clayton", 5.0)["beta"] > 0.3


def test_blncop_edge():
    out = blomqvists_beta_copula("gumbel", 2.0)
    assert out["beta"] == pytest.approx(4 * out["c_half"] - 1)
    with pytest.raises(ValueError):
        blomqvists_beta_copula("weibull", 2.0)
