"""Tests for ginicop."""

import numpy as np
import pytest

from morie.fn.ginicop import ginis_gamma_copula

def test_ginicop_basic():
    assert ginis_gamma_copula("independence")["gamma"] == pytest.approx(0.0, abs=1e-6)
    assert ginis_gamma_copula("gumbel", 5.0)["gamma"] > ginis_gamma_copula("gumbel", 1.5)["gamma"]


def test_ginicop_edge():
    with pytest.raises(ValueError):
        ginis_gamma_copula("clayton", 2.0, n=5)
