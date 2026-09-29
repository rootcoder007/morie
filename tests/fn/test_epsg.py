"""Tests for epsg.epsilon_greedy."""

from morie.fn import _array_core as np
from morie.fn.epsg import epsilon_greedy


def test_epsg_basic():
    """Test basic functionality."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_greedy(arms)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_epsg_edge():
    """Test edge cases."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_greedy(arms)
    assert isinstance(result, dict)


def test_van_der_corput_bandit_replayed():
    import pytest

    def vdc(i, base):
        f, r, k = 1.0, 0.0, i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        return r

    mu = [0.2, 0.8, 0.5]
    e, T = 0.3, 20
    q, cnt, tot = [0.0] * 3, [0.0] * 3, 0.0
    for t in range(T):
        a = min(int(vdc(t, 3) * 3), 2) if vdc(t, 2) < e else max(range(3), key=lambda j: (q[j], -j))
        cnt[a] += 1
        q[a] += (mu[a] - q[a]) / cnt[a]
        tot += mu[a]
    r = epsilon_greedy(mu, epsilon=e, T=T)
    assert r["counts"] == cnt
    assert r["q"] == pytest.approx(q, rel=1e-14)
    assert r["estimate"] == pytest.approx(tot / T, rel=1e-14)
    assert r["p_greedy"] == pytest.approx(1 - e + e / 3, rel=1e-15)
