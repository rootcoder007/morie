"""Tests for hsirt."""

from morie.fn import _array_core as np
import pytest

from morie.fn._stats_core import norm

from morie.fn.hsirt import heteroskedastic_irt


def test_hsirt_basic():
    rng = np.random.default_rng(42)
    n, q = 25, 70
    x = np.linspace(-2, 2, n)
    beta = rng.normal(scale=1.2, size=q)
    alpha = rng.normal(scale=0.5, size=q)
    psi = np.ones(n)
    psi[0] = 4.0
    P = norm.cdf((x[:, None] * beta - alpha) / psi[:, None])
    V = (rng.random((n, q)) < P).astype(float)
    out = heteroskedastic_irt(V, x, item_params=(alpha, beta), max_iter=8)
    assert np.argmax(out["psi"]) == 0  # the unpredictable voter


def test_hsirt_edge():
    with pytest.raises(ValueError):
        heteroskedastic_irt(np.ones((4, 3)) * 2, np.zeros(4))  # non-binary
    with pytest.raises(ValueError):
        heteroskedastic_irt(np.ones((4, 3)), np.zeros(2))  # length mismatch
