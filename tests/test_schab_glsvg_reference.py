"""Cressie GLS semivariogram fit (eqs 4.30-4.32) and the drift bias (eq 5.35) against R."""

import math

from morie.fn._schab_glsvg import lag_pairs, matheron_covariance
from morie.fn.spglsv import gls_semivariogram_fit
from morie.fn.vgdrift import drift_semivariogram_bias

N = 36
CO = [(10 * ((i * 0.618034) % 1), 10 * ((i * 0.414214 + 0.3) % 1)) for i in range(N)]
Z = [
    math.sin(1.1 * x) + math.cos(0.9 * y) + math.sin(0.7 * x + 1.3 * y) + 0.3 * math.sin(7.1 * i)
    for i, (x, y) in enumerate(CO)
]
BR = [0, 0.8, 1.6, 2.4, 3.2, 4.0, 5.0, 6.0]


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_cressie_covariance_equals_trace_formula():
    pts, cls, _ = lag_pairs(CO, BR)
    R = matheron_covariance(pts, cls, 0.2, 2.0, 5.0, "exponential")
    # 2 tr(A_i Sigma A_j Sigma) with gamma_hat(h) = Z' A(h) Z
    diag = [
        0.29962184631112182,
        0.11615419706154145,
        0.26416136940204921,
        0.36320271149342753,
        0.40353978660983908,
        0.49314340113395816,
        0.54969891769341506,
    ]
    row1 = [
        0.29962184631112182,
        0.028774211115752343,
        0.067566773046412426,
        0.060439906183633568,
        0.057427907166929748,
        0.062522130860420813,
        0.053651774335276646,
    ]
    assert all(close(R[i][i], diag[i], 1e-12) for i in range(7))
    assert all(close(a, b, 1e-12) for a, b in zip(R[0], row1))


def test_gls_fit_matches_optim_loop():
    f = gls_semivariogram_fit(CO, Z, BR, "exponential")
    gh = [
        0.27569486041069963,
        0.86082657910209071,
        2.4608017740039405,
        2.458627367132832,
        1.7760948942595789,
        1.9238149595719074,
        2.0367351887342777,
    ]
    assert all(close(a, b, 1e-14) for a, b in zip(f["gamma_hat"], gh))
    assert close(f["sill"], 2.2182678263156599, 2e-6)
    assert close(f["range"], 6.3002832224103127, 2e-6)
    assert f["nugget"] < 1e-10
    assert close(f["criterion"], 26.695525008551186, 1e-6)


def test_drift_expectation_matches_sigma_route():
    X = [[1, x, y] for x, y in CO]
    d = drift_semivariogram_bias(CO, X, [2, 0.3, -0.2], BR, 0.1, 1.0, 4.0, "exponential")
    ref = [
        0.57541916978473129,
        0.67803046303768688,
        0.96757852460361293,
        1.3400351558026318,
        1.3936400637999211,
        1.5497724270514073,
        2.196069585434957,
    ]
    assert all(close(a, b, 1e-13) for a, b in zip(d["expected"], ref))
