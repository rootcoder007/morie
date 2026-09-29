"""Tests for ksr07.kosorok_bootstrap_empirical."""

from morie.fn import _array_core as np
from morie.fn.ksr07 import kosorok_bootstrap_empirical


def test_ksr07_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = kosorok_bootstrap_empirical(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_ksr07_edge():
    """Test edge cases."""
    x = np.array([1.0, 2.0])
    result = kosorok_bootstrap_empirical(x)
    assert result["n"] == 2


def test_multinomial_bootstrap_replayed():
    import math

    import pytest

    from morie.fn._tail1core import Lcg

    x = [1.0, 2.5, 3.0, 4.5, 7.0]
    n, B = 5, 12
    g = Lcg(9)
    st = []
    for _ in range(B):
        st.append(sum(x[min(int(g.unif() * n), n - 1)] for _ in range(n)) / n)
    m = sum(st) / B
    sd = math.sqrt(sum((v - m) ** 2 for v in st) / (B - 1))
    r = kosorok_bootstrap_empirical(x, B=B, seed=9)
    assert r["boot_mean"] == pytest.approx(m, rel=1e-13)
    assert r["process_sd"] == pytest.approx(math.sqrt(n) * sd, rel=1e-12)
    q = sorted(st)
    assert (r["ci_lower"], r["ci_upper"]) == (q[math.floor(0.025 * (B - 1))], q[math.ceil(0.975 * (B - 1))])
