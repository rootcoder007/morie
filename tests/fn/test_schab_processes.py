"""Point-process models, Ch. 3. Assertions are the book's properties.

Schabenberger & Gotway (2005), Secs 3.2, 3.3, 3.7.2.
"""

import numpy as np
import pytest

from morie.fn.sppois import schabenberger_poisson_process as hpp
from morie.fn.spbino import schabenberger_binomial_process as binomial
from morie.fn.spcsr import schabenberger_csr_def as csr
from morie.fn.spnscl import schabenberger_neyman_scott as neyman_scott
from morie.fn.spthom import schabenberger_thomas_process as thomas

REGION = (0.0, 0.0, 10.0, 10.0)


def test_hpp_count_has_mean_equal_to_variance():
    """N(A) ~ Poisson(lambda nu(A)); a Poisson has mean = variance."""
    lam = 2.0
    counts = [hpp(lam, REGION, seed=s)["n"] for s in range(400)]
    expected = lam * 100.0
    assert np.mean(counts) == pytest.approx(expected, rel=0.05)
    assert np.var(counts) == pytest.approx(expected, rel=0.20)


def test_hpp_reports_the_theoretical_moments():
    r = hpp(3.0, REGION, seed=1)
    assert r["expected_n"] == pytest.approx(300.0)
    assert r["var_n"] == pytest.approx(r["expected_n"])   # Poisson


def test_binomial_count_is_fixed_not_random():
    """The defining difference from the Poisson process."""
    for s in range(5):
        assert binomial(150, REGION, seed=s)["n"] == 150


def test_binomial_subregion_variance_is_below_its_mean():
    """Var[N(B)] = np(1-p) < np, whereas a Poisson has them equal."""
    r = binomial(200, REGION, seed=0)
    assert r["binomial_var_half"] < r["binomial_mean_half"]
    assert r["binomial_var_half"] == pytest.approx(200 * 0.5 * 0.5)
    f = r["counts_in_fraction"]
    m, v = f(0.25)
    assert (m, v) == pytest.approx((50.0, 200 * 0.25 * 0.75))


def test_binomial_points_lie_inside_the_region():
    p = binomial(300, REGION, seed=2)["points"]
    assert np.all((p[:, 0] >= 0) & (p[:, 0] <= 10))
    assert np.all((p[:, 1] >= 0) & (p[:, 1] <= 10))


def test_csr_diagnostics_are_near_one_on_a_csr_pattern():
    pts = np.random.default_rng(0).random((800, 2)) * 10
    r = csr(pts, REGION)
    assert r["index_of_dispersion"] == pytest.approx(1.0, abs=0.35)
    assert r["clark_evans"] == pytest.approx(1.0, abs=0.10)


def test_clustering_raises_dispersion_and_lowers_clark_evans():
    rng = np.random.default_rng(0)
    parents = rng.random((40, 2)) * 10
    clustered = np.clip(np.repeat(parents, 20, axis=0)
                        + rng.normal(0, 0.15, (800, 2)), 0, 10)
    r = csr(clustered, REGION)
    assert r["index_of_dispersion"] > 2.0
    assert r["clark_evans"] < 0.8


def test_regularity_moves_the_diagnostics_the_other_way():
    rng = np.random.default_rng(1)
    g = np.linspace(0.5, 9.5, 28)
    pts = np.clip(np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
                  + rng.normal(0, 0.03, (784, 2)), 0, 10)
    r = csr(pts, REGION)
    assert r["index_of_dispersion"] < 1.0
    assert r["clark_evans"] > 1.0


def test_neyman_scott_k_exceeds_the_poisson_k_everywhere():
    """The clustering excess is strictly positive."""
    r = np.linspace(0.05, 2.0, 20)
    out = neyman_scott(r, rho=10.0, mu=5.0, sigma=0.1)
    assert np.all(out["k"] > out["k_csr"])
    assert np.all(out["excess"] > 0)


def test_neyman_scott_excess_matches_the_closed_form():
    """K(r) - pi r^2 = (1 - exp(-r^2 / 4 sigma^2)) / rho."""
    r = np.array([0.0, 0.25, 1.0])
    rho, sigma = 7.0, 0.2
    out = neyman_scott(r, rho=rho, mu=3.0, sigma=sigma)
    np.testing.assert_allclose(
        out["excess"], (1.0 - np.exp(-(r**2) / (4 * sigma**2))) / rho, rtol=1e-12)


def test_neyman_scott_excess_vanishes_as_parents_get_dense():
    """Many sparse clusters are indistinguishable from Poisson."""
    assert neyman_scott(0.5, rho=1e6, mu=5.0, sigma=0.1)["excess"][0] < 1e-5


def test_neyman_scott_intensity_is_rho_times_mu():
    assert neyman_scott(1.0, rho=4.0, mu=6.0, sigma=0.1)["lambda"] == \
        pytest.approx(24.0)


def test_thomas_is_the_gaussian_case_of_neyman_scott():
    r = np.linspace(0, 1.5, 10)
    np.testing.assert_allclose(thomas(r, 10.0, 5.0, 0.1)["k"],
                               neyman_scott(r, 10.0, 5.0, 0.1)["k"], rtol=1e-15)
    assert "k_function" in thomas(r, 10.0, 5.0, 0.1)


def test_process_input_validation():
    with pytest.raises(ValueError, match="`lam` must be"):
        hpp(0.0, REGION)
    with pytest.raises(ValueError, match="`n` must be"):
        binomial(-1, REGION)
    for bad in ({"rho": 0.0}, {"mu": 0.0}, {"sigma": 0.0}):
        with pytest.raises(ValueError, match="must all be"):
            neyman_scott(1.0, **{"rho": 1.0, "mu": 1.0, "sigma": 1.0, **bad})
    with pytest.raises(ValueError, match="non-negative"):
        neyman_scott(np.array([-1.0]))
