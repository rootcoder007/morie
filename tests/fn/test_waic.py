"""Tests for morie.fn.waic -- WAIC."""

from morie.fn import _array_core as np
from morie.fn.waic import compute_waic


def test_returns_dict():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (100, 20))
    result = compute_waic(ll)
    assert isinstance(result, dict)
    assert "waic" in result
    assert "lppd" in result
    assert "p_waic" in result


def test_pointwise_shape():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (50, 10))
    result = compute_waic(ll)
    assert len(result["pointwise_waic"]) == 10


def test_waic_equals_sum():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (50, 10))
    result = compute_waic(ll)
    np.testing.assert_allclose(
        result["waic"],
        -2 * (result["lppd"] - result["p_waic"]),
        atol=1e-8,
    )


def test_p_waic_nonnegative():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (50, 10))
    result = compute_waic(ll)
    assert result["p_waic"] >= 0


def test_invalid_1d():
    try:
        compute_waic([1.0, 2.0, 3.0])
        assert False
    except ValueError:
        pass


def test_too_few_samples():
    try:
        compute_waic([[1.0, 2.0]])
        assert False
    except ValueError:
        pass


def test_pointwise_waic_and_se_recomputed():
    import math

    import pytest

    ll = [[-1.0, -0.5, -2.0], [-1.2, -0.4, -1.5], [-0.8, -0.7, -2.5], [-1.1, -0.6, -1.9]]
    S = 4
    pw = []
    for i in range(3):
        col = [ll[s][i] for s in range(S)]
        m = sum(col) / S
        lp = math.log(sum(math.exp(v) for v in col) / S)
        pw.append(-2 * (lp - sum((v - m) ** 2 for v in col) / (S - 1)))
    mp = sum(pw) / 3
    r = compute_waic(ll)
    assert [float(v) for v in r["pointwise_waic"]] == pytest.approx(pw, rel=1e-13)
    assert r["se"] == pytest.approx(math.sqrt(3 * sum((v - mp) ** 2 for v in pw) / 2), rel=1e-12)
