"""Tests for ksr043 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr043 import kosorok_ch2_quantile_hadamard_inequality


def test_ksr043_basic():
    from scipy import stats
    out = kosorok_ch2_quantile_hadamard_inequality(
        stats.norm.cdf, lambda z: 0.1 * stats.norm.pdf(z), t_n=0.01, p=0.7)
    assert out["sandwich_holds"] is True


def test_ksr043_edge():
    from scipy import stats
    with pytest.raises(ValueError):
        kosorok_ch2_quantile_hadamard_inequality(stats.norm.cdf, lambda z: 0.0,
                                                 t_n=0.01, p=1.5)
