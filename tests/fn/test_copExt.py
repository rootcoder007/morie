"""Tests for copExt."""

from morie.fn import _array_core as np
import pytest

from morie.fn._copula import copula_cdf
from morie.fn.copExt import extremal_copula

def test_copExt_basic():
    c = extremal_copula(0.4, 0.7, "gumbel", 2.0)
    assert c["valid_pickands"] is True
    assert c["cdf"] == pytest.approx(copula_cdf("gumbel", 0.4, 0.7, 2.0))


def test_copExt_edge():
    # max-stability: C(u^k, v^k) = C(u, v)^k
    c = extremal_copula(0.4, 0.7, "galambos", 1.5)
    ck = extremal_copula(0.4**2, 0.7**2, "galambos", 1.5)
    assert ck["cdf"] == pytest.approx(c["cdf"] ** 2, rel=1e-8)
    with pytest.raises(ValueError):
        extremal_copula(0.0, 0.7, "gumbel", 2.0)
