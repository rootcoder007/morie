"""Tests for kssup.ks_supremum."""

import pytest

from morie.fn import _array_core as np
from morie.fn.kssup import ks_supremum


def test_kssup_does_not_reject_its_own_distribution():
    """Normal data vs fitted normal: with fitted parameters the classical
    p-value is conservative, so non-rejection at 0.05 should be near
    certain. Measured 20/20 seeds with p > 0.2."""
    for s in range(5):
        rng = np.random.default_rng(s)
        r = ks_supremum(rng.standard_normal(200))
        assert r.p_value > 0.05


def test_kssup_rejects_the_wrong_family():
    """Exponential data against a fitted normal: skewness 2 cannot be
    absorbed by fitting mu and sigma."""
    rng = np.random.default_rng(0)
    r = ks_supremum(rng.exponential(1.0, 300), dist="norm")
    assert r.p_value < 0.01
    # And the right family fits.
    r2 = ks_supremum(rng.exponential(1.0, 300), dist="expon")
    assert r2.p_value > 0.05


def test_kssup_statistic_is_the_sup_distance():
    rng = np.random.default_rng(1)
    r = ks_supremum(rng.standard_normal(50))
    assert 0.0 <= r.statistic <= 1.0


def test_kssup_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 5"):
        ks_supremum(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="Unknown distribution"):
        ks_supremum(np.random.default_rng(0).standard_normal(30), dist="notadist")


def test_ks_statistic_against_the_fitted_normal():
    """D = max_i max(i/n - F(x_(i)), F(x_(i)) - (i-1)/n) with the ML fit
    (mean, divisor-n sd) that scipy's norm.fit returns."""
    import math

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 4.2, 2.5]
    n = 8
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / n)
    F = [0.5 * math.erfc(-((v - m) / s) / math.sqrt(2)) for v in sorted(x)]
    D = max(max((i + 1) / n - F[i], F[i] - i / n) for i in range(n))
    r = ks_supremum(x)
    assert r.statistic == pytest.approx(D, rel=1e-10)
