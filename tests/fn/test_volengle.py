"""Tests for volengle.vol_engle_lagrange (ARCH-LM)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volengle import vol_engle_lagrange


def _garch(seed, n=1500, omega=0.1, alpha=0.3, beta=0.6):
    rng = np.random.default_rng(seed)
    s2, y = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        y[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * y[t] ** 2 + beta * s2
    return y


def test_volengle_detects_arch():
    """GARCH data must reject decisively at every reasonable lag order."""
    y = _garch(0)
    for q in (1, 4):
        r = vol_engle_lagrange(y, q=q)
        assert float(r["p_value"]) < 1e-4, f"q={q}"
        assert int(r["df"]) == q


def test_volengle_size_under_iid_noise():
    """i.i.d. Gaussian noise has no ARCH; measured 1/30 rejections at
    alpha = 0.05 across seeds (nominal 1.5)."""
    rej = 0
    for s in range(30):
        rng = np.random.default_rng(s)
        r = vol_engle_lagrange(rng.standard_normal(400), q=2)
        rej += float(r["p_value"]) < 0.05
    assert rej <= 5


def test_volengle_q_actually_changes_the_regression():
    """The placeholder ignored q entirely -- the statistic was identical
    for every lag order. The real LM statistic differs."""
    y = _garch(3, n=800)
    s1 = float(vol_engle_lagrange(y, q=1)["statistic"])
    s5 = float(vol_engle_lagrange(y, q=5)["statistic"])
    assert s1 != pytest.approx(s5, rel=1e-6)


def test_volengle_a_ks_test_would_miss_this():
    """LM is a conditional-heteroskedasticity test, not a normality test:
    it must NOT reject on i.i.d. data drawn from a heavy-tailed t(4),
    which a KS-vs-normal check would flag."""
    rng = np.random.default_rng(5)
    r = vol_engle_lagrange(rng.standard_t(4, 800), q=2)
    assert float(r["p_value"]) > 0.01


def test_volengle_rejects_bad_input():
    with pytest.raises(ValueError, match="q must be"):
        vol_engle_lagrange(np.arange(50.0), q=0)
    with pytest.raises(ValueError, match="at least"):
        vol_engle_lagrange(np.arange(3.0), q=2)
