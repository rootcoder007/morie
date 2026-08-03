"""Tests for the Ghosal Ch 2 function-space priors."""
import math

from morie.fn.gh_c2_1 import ghosal_random_basis_expansion
from morie.fn.gh_c2_2 import ghosal_gp_prior_def
from morie.fn.gh_c2_3 import ghosal_gp_increasing_prior
from morie.fn.gh_c2_4 import ghosal_exp_link
from morie.fn.gh_c2_5 import ghosal_histogram_prior
from morie.fn.gh_c2_6 import ghosal_mixture_basis_prior
from morie.fn.gh_c2_7 import ghosal_bernstein_feller
from morie.fn.gh_c2_8 import ghosal_np_normal_reg
from morie.fn.gh_c2_9 import ghosal_np_binary_reg
from morie.fn.gh_c2_10 import ghosal_np_poisson_reg

GRID = [i / 40.0 for i in range(41)]


def test_basis_expansion_reproducible_and_finite():
    a = ghosal_random_basis_expansion(GRID, seed=1)
    b = ghosal_random_basis_expansion(GRID, seed=1)
    assert a["f"] == b["f"]
    assert all(abs(v) < 50 for v in a["f"])


def test_gp_draw_covariance_scale():
    r = ghosal_gp_prior_def(GRID, var=1.0, seed=3)
    # a unit-variance GP path stays within +-5 sd on 41 points
    assert all(abs(v) < 5.0 for v in r["f"])


def test_increasing_process_is_increasing():
    r = ghosal_gp_increasing_prior(GRID, seed=4)
    assert r["increasing"] is True
    assert r["estimate"] > 0


def test_exp_link_normalizes():
    r = ghosal_exp_link(GRID)
    d = r["density"]
    Z = sum(0.5 * (d[i] + d[i - 1]) * (GRID[i] - GRID[i - 1])
            for i in range(1, len(GRID)))
    assert abs(Z - 1.0) < 1e-9
    assert min(d) > 0


def test_histogram_weights_simplex():
    r = ghosal_histogram_prior(GRID, K=8, seed=5)
    assert abs(sum(r["weights"]) - 1.0) < 1e-12
    Z = sum(v * (GRID[1] - GRID[0]) for v in r["density"][:-1])
    assert abs(Z - 1.0) < 0.06        # Riemann error only


def test_mixture_density_positive_weights_simplex():
    r = ghosal_mixture_basis_prior(GRID, seed=6)
    assert abs(sum(r["weights"]) - 1.0) < 1e-12
    assert min(r["density"]) >= 0


def test_bernstein_uniform_convergence():
    # F(x)=x^2: operator error is O(1/K) uniformly
    r30 = ghosal_bernstein_feller(GRID, K=30)
    r120 = ghosal_bernstein_feller(GRID, K=120)
    assert r120["sup_error"] < r30["sup_error"]
    assert r120["sup_error"] < 0.01


def test_gp_regression_recovers_signal():
    xs = [i / 20.0 for i in range(21)]
    ys = [math.sin(4 * v) for v in xs]
    r = ghosal_np_normal_reg(xs, ys, sigma2=1e-4)
    err = max(abs(a - b) for a, b in zip(r["fitted"], ys))
    assert err < 0.05


def test_binary_reg_separates_classes():
    xs = [i / 20.0 for i in range(21)]
    ys = [0.0 if v < 0.5 else 1.0 for v in xs]
    r = ghosal_np_binary_reg(xs, ys)
    assert r["prob"][0] < 0.35 and r["prob"][-1] > 0.65


def test_poisson_reg_tracks_intensity():
    xs = [i / 10.0 for i in range(11)]
    ys = [1.0, 1, 2, 2, 3, 4, 6, 8, 10, 13, 16]
    r = ghosal_np_poisson_reg(xs, ys)
    lam = r["intensity"]
    assert lam[-1] > 3 * lam[0]
    assert all(v > 0 for v in lam)
