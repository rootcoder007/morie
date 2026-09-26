"""Schabenberger & Gotway book methods against R references (solve, anova, mvtnorm, copula, lpSolve)."""

import math

from morie.fn.bvcchy import bivariate_cauchy_density
from morie.fn.csinv import compound_symmetry_inverse
from morie.fn.mgamrf import multivariate_gamma_field
from morie.fn.nestvc import nested_variance_components
from morie.fn.plackt import plackett_distribution
from morie.fn.rhobin import binary_equicorrelation_bound


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_csinv_matches_solve():
    r = compound_symmetry_inverse(4, 3, 0.7)
    assert close(r["diag"], 0.375106564364876383, 1e-14)
    assert close(r["offdiag"], -0.059676044330775758, 1e-14)
    assert not compound_symmetry_inverse(3, 2, 2)["exists"]
    assert not compound_symmetry_inverse(3, 1, -0.5)["exists"]


def test_nestvc_matches_nested_anova():
    fac = [1] * 15 + [2] * 15
    bat = [b for b in range(1, 7) for _ in range(5)]
    k = [j for _ in range(6) for j in range(1, 6)]
    eff = [-0.3, 0.5, 0.1, 0.25, -0.4, 0.2]
    y = [
        10 + 0.4 * f + eff[b - 1] + 0.1 * math.sin(1.7 * (i + 1)) + 0.05 * math.cos(3.1 * kk * b)
        for i, (f, b, kk) in enumerate(zip(fac, bat, k))
    ]
    r = nested_variance_components(y, fac, bat)
    # anova(lm(y ~ fac / bat)) mean squares
    assert close(r["ms_unit"], 0.8368069116642972149, 1e-12)
    assert close(r["ms_error"], 0.0071806039350631344, 1e-12)
    assert close(r["sigma2_unit"], (0.8368069116642972149 - 0.0071806039350631344) / 5, 1e-12)
    # nlme::lme(REML), to its convergence tolerance
    assert close(r["sigma2_unit"], 0.1659252609654789379, 1e-8)


def test_bvcchy_matches_dmvt():
    r = bivariate_cauchy_density([0, 0.5, 3], [0, -1.2, 2], 1.5)["density"]
    ref = [0.0707355302630646304, 0.0305258002453248876, 0.0040087283290591302]
    assert all(close(a, b, 1e-14) for a, b in zip(r, ref))


def test_plackett_matches_copula():
    r = plackett_distribution([0.2, 0.5, 0.9], [0.7, 0.5, 0.3], 3.5)
    ref = [0.17407983862836823, 0.32583426132260590, 0.28814064431721093]
    assert all(close(a, b, 1e-12) for a, b in zip(r["F12"], ref))
    assert close(r["rho"], 0.39690547528518771, 1e-12)
    assert close(plackett_distribution(0.3, 0.6, 0.4)["rho"], -0.29713170694632218, 1e-12)
    assert plackett_distribution(0.3, 0.6, 1)["F12"] == [0.3 * 0.6]


def test_rhobin_matches_lp():
    for mu, n, ref in [
        (0.3, 5, -0.19047619047619047),
        (0.5, 4, -0.33333333333333337),
        (0.37, 7, -0.14195828481542783),
        (0.1, 3, -0.11111111111111112),
    ]:
        assert close(binary_equicorrelation_bound(mu, n)["rho_lower"], ref, 1e-14)


def test_mgamrf_problem_2_3():
    r = multivariate_gamma_field([2, 1, 1, 3], 0.5)
    assert r["cov"] == [[0.75, 0.5, 0.5], [0.5, 0.75, 0.5], [0.5, 0.5, 1.25]]
    assert close(r["corr"][0][2], 2 / math.sqrt(15), 1e-15)
    assert r["shape"] == [3, 3, 5] and not r["stationary"]
    assert multivariate_gamma_field([2, 1, 1], 0.5)["stationary"]
