"""Tests for ksr022 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr022 import kosorok_ch1_multiplicative_intensity


def test_ksr022_basic():
    rng = np.random.default_rng(1)
    n = 400
    Z = rng.standard_normal((n, 1))
    T = rng.exponential(1.0 / np.exp(Z[:, 0] * 0.5))
    C = rng.exponential(3.0, n)
    out = kosorok_ch1_multiplicative_intensity(np.minimum(T, C),
                                               (T <= C).astype(float), Z, beta=[0.5])
    assert np.all(np.diff(out["cumulative_hazard"]) >= 0)  # monotone Breslow


def test_ksr022_edge():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((50, 1))
    with pytest.raises(ValueError):
        kosorok_ch1_multiplicative_intensity(np.ones(50), np.zeros(50), Z)  # no events
