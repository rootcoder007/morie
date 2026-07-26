"""antth: antithetic-variate Monte Carlo (Hammersley & Morton 1956)."""

import numpy as np
import pytest

from morie.fn.antth import antithetic_variates as av


def test_antth_default_integrand_recovers_one_half():
    """f(u) = u on (0,1) has E = 1/2, and antithetic pairing makes it EXACT:
    (u + (1-u))/2 = 1/2 for every draw, so the variance is zero."""
    r = av(N=1000, seed=1)
    assert r["estimate"] == pytest.approx(0.5, abs=1e-12)
    assert r["se"] == pytest.approx(0.0, abs=1e-12)


def test_antth_beats_crude_monte_carlo_on_a_monotone_integrand():
    """The whole justification: for monotone f, u and 1-u are negatively
    correlated, so the paired average has lower variance."""
    r = av(f=lambda u: np.exp(u), N=20_000, seed=3)
    assert r["var_ratio_av_over_crude"] < 1.0


def test_antth_estimates_a_known_integral():
    """int_0^1 exp(u) du = e - 1 = 1.718281828..."""
    r = av(f=lambda u: np.exp(u), N=50_000, seed=5)
    assert r["estimate"] == pytest.approx(np.e - 1, rel=1e-3)


def test_antth_is_exact_for_any_linear_integrand():
    """Antithetic pairing cancels the linear part exactly, whatever the
    slope and intercept."""
    for a, b in ((3.0, -1.0), (0.5, 2.0)):
        r = av(f=lambda u, a=a, b=b: a * u + b, N=500, seed=7)
        assert r["estimate"] == pytest.approx(a * 0.5 + b, abs=1e-12)


def test_antth_reports_the_crude_estimate_for_comparison():
    r = av(f=lambda u: u**2, N=10_000, seed=11)
    assert np.isfinite(r["estimate_crude"])
    assert r["estimate"] == pytest.approx(1 / 3, rel=0.02)
    # N counts PAIRS, not total draws: N=10_000 means 10,000 antithetic
    # pairs and so 20,000 function evaluations.
    assert r["n_pairs"] == 10_000


def test_antth_gives_no_benefit_on_a_symmetric_integrand():
    """Honest limitation, and the boundary case of the method.

    For f symmetric about u = 1/2 we have f(u) = f(1-u) exactly, so the
    antithetic average IS f(u) and the pairing buys nothing at all: the
    variance ratio is exactly 1.0, not below it. Antithetic sampling helps
    for monotone integrands and is wasted here -- which is why the ratio is
    reported rather than assumed.
    """
    r = av(f=lambda u: (u - 0.5) ** 2, N=20_000, seed=13)
    assert r["var_ratio_av_over_crude"] == pytest.approx(1.0, rel=1e-9)
    # ...whereas a monotone integrand on the same budget does gain.
    mono = av(f=lambda u: np.exp(u), N=20_000, seed=13)
    assert mono["var_ratio_av_over_crude"] < 1.0


def test_antth_is_reproducible_for_a_fixed_seed():
    a = av(f=lambda u: np.sqrt(u), N=1000, seed=17)["estimate"]
    b = av(f=lambda u: np.sqrt(u), N=1000, seed=17)["estimate"]
    assert a == pytest.approx(b, abs=0.0)
