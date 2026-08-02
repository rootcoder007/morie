"""Granger/info-flow cluster: ggrcst, granci, trnfen."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ggrcst import granger_causality
from morie.fn.granci import granger_causality_info
from morie.fn.trnfen import transfer_entropy


def _coupled(seed, n=1500, beta=0.6):
    # x drives y with one lag; y does not drive x.
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    ex = rng.normal(size=n)
    ey = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.4 * y[t - 1] + beta * x[t - 1] + ey[t]
    return x, y


def test_granger_detects_direction():
    rejects_fwd = rejects_rev = 0
    for seed in range(10):
        x, y = _coupled(seed)
        rejects_fwd += granger_causality(x, y, p=1)["p_value"] < 0.01
        rejects_rev += granger_causality(y, x, p=1)["p_value"] < 0.01
    assert rejects_fwd == 10  # beta = 0.6 at n = 1500: power ~ 1
    assert rejects_rev <= 2  # measured 0/10; size ~ alpha


def test_granger_null_size():
    rejects = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=800)
        y = rng.normal(size=800)
        rejects += granger_causality(x, y, p=2)["p_value"] < 0.05
    assert rejects <= 4  # measured 1/20 at alpha = 0.05


def test_granger_validation():
    with pytest.raises(ValueError):
        granger_causality([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], p=1)  # too short
    with pytest.raises(ValueError):
        granger_causality([1.0], [1.0, 2.0], p=1)  # length mismatch


def test_granci_equals_half_log_rss_ratio():
    x, y = _coupled(0)
    g = granger_causality(x, y, p=1)
    ci = granger_causality_info(x, y, lag=1)
    assert ci["mi"] == pytest.approx(0.5 * np.log(g["rss_restricted"] / g["rss_unrestricted"]))
    assert ci["p_value"] < 0.01
    # no information in the null direction: measured mi ~ 5e-4
    assert granger_causality_info(y, x, lag=1)["mi"] < 0.01


def test_trnfen_gaussian_equals_granci():
    x, y = _coupled(3)
    te = transfer_entropy(x, y, lag=1, method="gaussian")
    ci = granger_causality_info(x, y, lag=1)
    assert te["te"] == pytest.approx(ci["mi"])


def test_trnfen_binned_directionality():
    fwd_wins = 0
    for seed in range(8):
        x, y = _coupled(seed, n=4000, beta=0.8)
        te_fwd = transfer_entropy(x, y, method="binned", bins=4)["te"]
        te_rev = transfer_entropy(y, x, method="binned", bins=4)["te"]
        fwd_wins += te_fwd > te_rev
        assert te_fwd >= 0.0
    assert fwd_wins >= 7  # measured 8/8
    with pytest.raises(ValueError):
        transfer_entropy([1.0] * 20, [1.0] * 20, method="binned", bins=4)  # too short
    with pytest.raises(ValueError):
        transfer_entropy([1.0] * 100, [1.0] * 100, method="nope")
