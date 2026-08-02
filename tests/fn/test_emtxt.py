"""Tests for emtxt."""

from morie.fn import _array_core as np
import pytest

from morie.fn.emtxt import em_irt_text


def test_emtxt_basic():
    rng = np.random.default_rng(42)
    n, k = 8, 50
    theta = np.linspace(-1.5, 1.5, n)
    psi = rng.normal(1.0, 0.3, size=k)
    beta = rng.normal(0.0, 0.8, size=k)
    Y = rng.poisson(np.exp(1.0 + psi[None, :] + beta[None, :] * theta[:, None]))
    out = em_irt_text(Y, polarity=(0, n - 1))
    assert np.corrcoef(out["theta"], theta)[0, 1] > 0.95
    assert out["theta"][0] < out["theta"][-1]


def test_emtxt_edge():
    with pytest.raises(ValueError):
        em_irt_text(np.full((5, 5), -1.0))  # negative counts
    with pytest.raises(ValueError):
        em_irt_text(np.ones((5, 5)), polarity=(0, 0))
