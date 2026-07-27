"""Tests for copjoe."""

import numpy as np
import pytest

from morie.fn.copjoe import joe_copula

def test_copjoe_basic():
    out = joe_copula(0.5, 0.5, 2.5)
    assert 0 < out["tau"] < 1
    assert out["cdf"] < 0.5


def test_copjoe_edge():
    assert joe_copula(0.4, 0.6, 1.0)["tau"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        joe_copula(0.5, 0.5, 0.5)
