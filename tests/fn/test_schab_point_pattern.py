"""Book-certified tests for the Schabenberger & Gotway point-pattern functions.

The reference values are theoretical, not self-generated: under CSR the
book gives K(h) = pi h^2 exactly, and the nearest-neighbour and empty
space distributions are both 1 - exp(-lambda pi r^2). Departures are
asserted in the DIRECTIONS the book states for clustered and regular
patterns.

Schabenberger, O. & Gotway, C. A. (2005). Ch. 3.
"""

import numpy as np
import pytest

from morie.fn.spkfun import schabenberger_k_function as k_function
from morie.fn.splfun import schabenberger_l_function as l_function
from morie.fn.spgfun import schabenberger_g_function as g_function
from morie.fn.spffun import schabenberger_f_function as f_function

REGION = (0.0, 0.0, 10.0, 10.0)
R = np.linspace(0.1, 1.5, 8)


def _csr(seed=0, n=800):
    return np.random.default_rng(seed).random((n, 2)) * 10.0


def _clustered(seed=1):
    rng = np.random.default_rng(seed)
    parents = rng.random((40, 2)) * 10.0
    return np.clip(np.repeat(parents, 20, axis=0)
                   + rng.normal(0, 0.15, (800, 2)), 0, 10)


def _regular(seed=1):
    rng = np.random.default_rng(seed)
    g = np.linspace(0.5, 9.5, 28)
    mesh = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    return np.clip(mesh + rng.normal(0, 0.03, mesh.shape), 0, 10), g[1] - g[0]


def test_k_function_of_a_csr_pattern_tracks_pi_h_squared():
    """K(h) = pi h^2 for the homogeneous Poisson process (p. 101)."""
    r = k_function(_csr(), r=R, region=REGION)
    np.testing.assert_allclose(r["k"], r["k_csr"], rtol=0.15)


def test_k_csr_reference_curve_is_exactly_pi_r_squared():
    r = k_function(_csr(), r=R, region=REGION)
    np.testing.assert_allclose(r["k_csr"], np.pi * R**2, rtol=1e-12)


def test_clustering_raises_k_above_the_csr_curve():
    """"In a clustered pattern ... the number of extra events within
    small distances will be large" (p. 101)."""
    r = k_function(_clustered(), r=R, region=REGION)
    assert np.all(r["k"] > r["k_csr"])


def test_regularity_lowers_k_below_the_csr_curve_at_short_lags():
    """"In regular patterns the number of extra events for short
    distances will be small" (p. 101).

    "Short" means shorter than the inter-point spacing; past that a
    lattice is not sparse any more.
    """
    pts, spacing = _regular()
    r = np.linspace(0.05, 0.8 * spacing, 6)
    got = k_function(pts, r=r, region=REGION)
    assert np.all(got["k"] < got["k_csr"])


def test_l_function_is_the_square_root_of_k_over_pi():
    """L(h) = sqrt(K(h)/pi) (p. 103)."""
    pts = _csr()
    kk = k_function(pts, r=R, region=REGION)["k"]
    ll = l_function(pts, r=R, region=REGION)["l"]
    np.testing.assert_allclose(ll, np.sqrt(np.maximum(kk, 0) / np.pi), rtol=1e-12)


def test_l_minus_r_is_the_csr_reference_line_at_zero():
    """Under CSR, K = pi h^2 so L(h) = h and L(h) - h = 0 (p. 103)."""
    r = l_function(_csr(), r=R, region=REGION)
    assert np.max(np.abs(r["l_minus_r"])) < 0.05


def test_clustering_pushes_l_minus_r_positive_at_short_distances():
    """"Clustering of events manifests itself as positive values at
    short distances" (p. 103)."""
    r = l_function(_clustered(), r=R, region=REGION)
    assert np.all(r["l_minus_r"] > 0)


def test_g_function_is_the_empirical_nn_cdf():
    """G_hat(y0) = #(y_i <= y0) / n (p. 97)."""
    pts = _csr(seed=3, n=200)
    r = g_function(pts, r=R, region=REGION)
    nn = r["nn_distances"]
    expected = np.array([(nn <= y).sum() / nn.size for y in r["r"]])
    np.testing.assert_allclose(r["g"], expected, rtol=1e-12)
    assert np.all(np.diff(r["g"]) >= 0)          # a CDF is non-decreasing


def test_g_matches_the_csr_form_on_a_csr_pattern():
    """Under CSR, G(y) = 1 - exp(-lambda pi y^2)."""
    r = g_function(_csr(), r=R, region=REGION)
    assert np.max(np.abs(r["g"] - r["g_csr"])) < 0.06


def test_clustering_lifts_g_and_depresses_f():
    """G and F respond to clustering in OPPOSITE directions: neighbours
    are closer, but the empty gaps between clusters are larger."""
    pts = _clustered()
    g = g_function(pts, r=R, region=REGION)
    f = f_function(pts, REGION, R)
    assert np.all(g["g"] >= g["g_csr"])
    assert np.all(f["f"] <= f["f_csr"])


def test_f_matches_the_csr_form_on_a_csr_pattern():
    r = f_function(_csr(), REGION, R)
    assert np.max(np.abs(r["f"] - r["f_csr"])) < 0.06


def test_f_is_a_non_decreasing_cdf():
    r = f_function(_csr(), REGION, R)
    assert np.all(np.diff(r["f"]) >= 0)
    assert np.all((r["f"] >= 0) & (r["f"] <= 1))


def test_intensity_is_count_over_area():
    """lambda_hat = N(A) / nu(A), eq (3.8)."""
    pts = _csr(n=500)
    r = k_function(pts, r=R, region=REGION)
    assert r["lambda_est"] == pytest.approx(500 / 100.0)


def test_border_correction_differs_from_the_naive_estimator():
    """The naive estimator is negatively biased near the boundary (p. 102)."""
    pts = _csr()
    a = k_function(pts, r=R, region=REGION, correction="border")["k"]
    b = k_function(pts, r=R, region=REGION, correction="none")["k"]
    assert not np.allclose(a, b)
    assert np.all(b[-3:] <= a[-3:] + 1e-12)      # naive is the lower one


def test_point_pattern_input_validation():
    with pytest.raises(ValueError, match="at least two events"):
        g_function(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        k_function(_csr(n=50), r=np.array([-1.0]), region=REGION)
    with pytest.raises(ValueError, match="positive area"):
        k_function(_csr(n=50), r=R, region=(0.0, 0.0, 0.0, 5.0))
