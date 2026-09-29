"""Tests for morie.fn.bloos -- PSIS-LOO."""

from morie.fn import _array_core as np
from morie.fn.bloos import psis_loo


def test_returns_dict():
    ll = -0.5 * np.random.default_rng(42).standard_normal((50, 20)) ** 2
    result = psis_loo(ll)
    assert isinstance(result, dict)
    assert "elpd_loo" in result


def test_looic_finite():
    ll = -0.5 * np.random.default_rng(42).standard_normal((100, 30)) ** 2
    result = psis_loo(ll)
    assert np.isfinite(result["looic"])


def test_k_hat_length():
    ll = -0.5 * np.random.default_rng(42).standard_normal((50, 10)) ** 2
    result = psis_loo(ll)
    assert len(result["k_hat"]) == 10


def _normal_ll(S, ys):
    import math

    mu = [-1.0 + 2.0 * s / (S - 1) for s in range(S)]
    return [[-0.5 * math.log(2 * math.pi * 0.49) - (y - m) ** 2 / (2 * 0.49) for y in ys] for m in mu]


def test_short_tail_is_plain_importance_sampling():
    """S = 20 gives a tail of ceil(min(4, 13.4)) = 4 < 5 draws: no Pareto fit,
    so elpd_i is the harmonic-mean identity -log mean_s exp(-ll_s)."""
    import math

    import pytest

    ll = _normal_ll(20, [0.3, -0.8, 2.4])
    r = psis_loo(ll)
    for i in range(3):
        col = [row[i] for row in ll]
        want = -math.log(sum(math.exp(-v) for v in col) / 20)
        assert r["elpd_loo_pointwise"][i] == pytest.approx(want, rel=1e-12)
        assert math.isinf(r["k_hat"][i])


def test_psis_identities():
    import math

    import pytest

    ll = _normal_ll(60, [0.3, -0.8, 2.4, 0.1])
    r = psis_loo(ll)
    lppd = sum(math.log(sum(math.exp(row[i]) for row in ll) / 60) for i in range(4))
    assert r["p_loo"] == pytest.approx(lppd - r["elpd_loo"], rel=1e-12)
    assert r["looic"] == pytest.approx(-2 * r["elpd_loo"], rel=1e-14)
    e = r["elpd_loo_pointwise"]
    m = sum(e) / 4
    assert r["se"] == pytest.approx(math.sqrt(4 * sum((v - m) ** 2 for v in e) / 3), rel=1e-12)
    assert all(math.isfinite(k) for k in r["k_hat"])
