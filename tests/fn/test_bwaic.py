"""Tests for morie.fn.bwaic -- WAIC."""

from morie.fn import _array_core as np
from morie.fn.bwaic import compute_waic


def test_returns_dict():
    ll = np.random.default_rng(42).standard_normal((50, 20))
    result = compute_waic(ll)
    assert isinstance(result, dict)
    assert "waic" in result


def test_waic_finite():
    ll = -0.5 * np.random.default_rng(42).standard_normal((50, 20)) ** 2
    result = compute_waic(ll)
    assert np.isfinite(result["waic"])


def test_p_waic_non_negative():
    ll = -0.5 * np.random.default_rng(42).standard_normal((100, 30)) ** 2
    result = compute_waic(ll)
    assert result["p_waic"] >= 0


def test_waic_terms_recomputed():
    import math

    import pytest

    ll = [[-1.0, -0.5, -2.0], [-1.2, -0.4, -1.5], [-0.8, -0.7, -2.5], [-1.1, -0.6, -1.9]]
    S, n = 4, 3
    lppd_i = [math.log(sum(math.exp(ll[s][i]) for s in range(S)) / S) for i in range(n)]
    pw_i = []
    for i in range(n):
        col = [ll[s][i] for s in range(S)]
        m = sum(col) / S
        pw_i.append(sum((v - m) ** 2 for v in col) / (S - 1))
    e = [a - b for a, b in zip(lppd_i, pw_i)]
    me = sum(e) / n
    r = compute_waic(ll)
    assert r["lppd"] == pytest.approx(sum(lppd_i), rel=1e-13)
    assert r["p_waic"] == pytest.approx(sum(pw_i), rel=1e-13)
    assert r["waic"] == pytest.approx(-2 * sum(e), rel=1e-13)
    assert r["se"] == pytest.approx(2 * math.sqrt(n * sum((v - me) ** 2 for v in e) / (n - 1)), rel=1e-12)
