"""ksr03: Glivenko-Cantelli / KS sup|F_n - F| statistic.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*, Ch. 2 -- in the library, filed under its ISBN
(978-0-387-74978-5) rather than its title.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr03 import kosorok_glivenko_cantelli as gc


def test_ksr03_statistic_vanishes_as_n_grows():
    """Glivenko-Cantelli itself: sup|F_n - F| -> 0 almost surely. That is the
    theorem this module exists to exhibit, so the test is the theorem.
    """
    rng = np.random.default_rng(2601)
    stats_ = [gc(rng.standard_normal(n))["statistic"] for n in (50, 500, 5000, 50_000)]
    assert stats_ == sorted(stats_, reverse=True)
    assert stats_[-1] < 0.02


def test_ksr03_agrees_with_scipy_kstest():
    from morie.fn import _stats_core as st

    rng = np.random.default_rng(2609)
    x = rng.standard_normal(300)
    ref = st.kstest(x, "norm")
    got = gc(x)
    assert got["statistic"] == pytest.approx(ref.statistic, rel=1e-9)
    assert got["p_value"] == pytest.approx(ref.pvalue, rel=1e-6)


def test_ksr03_rejects_a_badly_wrong_distribution():
    rng = np.random.default_rng(2617)
    x = rng.uniform(-3, 3, 500)
    assert gc(x, cdf="norm")["p_value"] < 1e-6


def test_ksr03_does_not_reject_the_true_distribution():
    rng = np.random.default_rng(2621)
    assert gc(rng.standard_normal(500), cdf="norm")["p_value"] > 0.05


def test_ksr03_accepts_a_callable_cdf():
    rng = np.random.default_rng(2633)
    x = rng.uniform(0, 1, 400)
    by_name = gc(x, cdf="uniform")["statistic"]
    by_call = gc(x, cdf=lambda v: np.clip(v, 0, 1))["statistic"]
    assert by_call == pytest.approx(by_name, rel=1e-9)


def test_ksr03_statistic_is_scaled_to_the_unit_interval():
    """sup|F_n - F| is a difference of two CDFs, so it cannot exceed 1."""
    rng = np.random.default_rng(2647)
    for n in (10, 100, 1000):
        s = gc(rng.standard_normal(n))["statistic"]
        assert 0.0 <= s <= 1.0


def test_ksr03_reports_the_sample_size():
    assert gc(np.random.default_rng(2657).standard_normal(77))["n"] == 77
