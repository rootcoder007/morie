"""Tests for the 9 closing Ghosal modules."""
import math

from morie.fn.ghs002 import ghosal_ch2_random_basis_expansion
from morie.fn.ghs003 import ghosal_ch2_basis_truncation_error
from morie.fn.ghs004 import ghosal_ch2_exponential_link_density
from morie.fn.ghs005 import ghosal_ch2_location_scale_mixture_limit
from morie.fn.ghs006 import ghosal_ch2_feller_density_approximation
from morie.fn.ghs007 import ghosal_ch2_binary_regression_density
from morie.fn.gh_hier_np import ghosal_hierarchical_np
from morie.fn.gh_loc_dp_crt import ghosal_local_dp_rate
from morie.fn.gh_var_dp_post import ghosal_variational_dp_posterior


def test_basis_expansion_hand_value():
    # single coefficient: f(x) = b sqrt(2) cos(pi x)
    r = ghosal_ch2_random_basis_expansion([2.0], x=0.0)
    assert abs(r["estimate"] - 2.0 * math.sqrt(2.0)) < 1e-12


def test_truncation_error_within_order():
    r = ghosal_ch2_basis_truncation_error(J=8)
    assert r["within_order"] is True
    bigger = ghosal_ch2_basis_truncation_error(J=2)
    assert bigger["estimate"] >= r["estimate"] - 1e-12


def test_exp_link_normalizes():
    # f = 0: uniform density 1 everywhere
    r = ghosal_ch2_exponential_link_density([0.0], x=0.3)
    assert abs(r["estimate"] - 1.0) < 1e-9
    r2 = ghosal_ch2_exponential_link_density([0.5], x=0.0)
    assert r2["estimate"] > 1.0          # boosted where f is large


def test_mixture_limit_converges():
    r = ghosal_ch2_location_scale_mixture_limit()
    assert r["converging"] is True
    assert r["estimate"] < 0.05


def test_feller_recovers_density():
    r = ghosal_ch2_feller_density_approximation()
    assert r["gap"] < 0.06               # density 2x at x=0.4 is 0.8


def test_binary_regression_likelihood():
    r = ghosal_ch2_binary_regression_density([1, 0], f=0.0)
    assert abs(r["estimate"] - 0.25) < 1e-12
    assert abs(r["success_prob"] - 0.5) < 1e-12


def test_hierarchical_alpha_posterior():
    r = ghosal_hierarchical_np()
    assert r["posterior_positive"] is True
    assert 0.2 < r["estimate"] < 15.0


def test_local_dp_rate_decreasing():
    r = ghosal_local_dp_rate()
    assert r["decreasing"] is True
    assert abs(r["exponent"] - 1.0 / 3.0) < 1e-12


def test_variational_dp_centers_split():
    r = ghosal_variational_dp_posterior()
    cs = sorted(r["centers"])
    assert min(abs(c + 2.0) for c in cs) < 0.3
    assert min(abs(c - 2.0) for c in cs) < 0.3
