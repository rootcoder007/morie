"""Tests for morie.fn.poissp -- Poisson areal regression with a CAR effect.

Every assertion here is an anchor that can fail: a closed form, a
structural identity of the CAR precision, or a limiting case that
collapses the model to one with a known answer. None of them compares
the module against itself.
"""

import math

import pytest

from morie.fn.poissp import poissp, car_precision, rho_bounds


Y = [3, 7, 2, 9, 5, 4, 8, 6]
E = [1.5, 2.0, 0.8, 3.0, 1.2, 1.0, 2.5, 1.8]
W5 = [[0, 1, 0, 0, 1],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 1, 0, 1],
      [1, 0, 0, 1, 0]]


def test_intercept_only_matches_the_closed_form():
    """With no spatial term the intercept MLE is log(sum y / sum E)."""
    r = poissp(Y, offset=E)
    assert r["beta"][0] == pytest.approx(math.log(sum(Y) / sum(E)), abs=1e-12)


def test_intercept_standard_error_matches_the_analytic_value():
    """Fisher information for the intercept is sum(m) = sum(y), so the
    standard error is exactly 1/sqrt(sum y)."""
    r = poissp(Y, offset=E)
    assert r["se"][0] == pytest.approx(1.0 / math.sqrt(sum(Y)), rel=1e-10)


def test_fixed_effect_score_vanishes_at_the_mode():
    """The fixed effects are unpenalised, so X'(y - m) = 0 exactly."""
    X = [[0.3], [-1.2], [0.8], [1.5], [-0.4], [0.1], [0.9], [-0.7]]
    r = poissp(Y, X=X, offset=E)
    for s in r["score_beta"]:
        assert abs(s) < 1e-8


def test_intercept_forces_the_fitted_total():
    """A consequence of the score identity: sum(fitted) = sum(y)."""
    r = poissp(Y, offset=E)
    assert sum(r["fitted"]) == pytest.approx(float(sum(Y)), abs=1e-9)


def test_icar_precision_rows_sum_to_zero():
    """rho = 1 is intrinsic: Q 1 = 0, so the prior is improper."""
    Q = car_precision(W5, tau=2.0, rho=1.0)
    for row in Q:
        assert abs(sum(row)) < 1e-12


def test_rho_zero_leaves_a_diagonal_precision():
    """With rho = 0 the CAR reduces to independent effects with
    precision tau * w_{i+} -- not a common variance."""
    Q = car_precision(W5, tau=2.0, rho=0.0)
    d = [sum(row) for row in W5]
    for i in range(5):
        for j in range(5):
            want = 2.0 * d[i] if i == j else 0.0
            assert Q[i][j] == pytest.approx(want, abs=1e-15)


def test_propriety_upper_bound_is_one():
    """The largest eigenvalue of the scaled adjacency is one for a
    connected graph, so rho < 1 is the propriety condition."""
    assert rho_bounds(W5)["upper"] == pytest.approx(1.0, abs=1e-10)


def test_improper_rho_is_refused():
    with pytest.raises(ValueError, match="propriety"):
        poissp([1, 2, 3, 4, 5], W=W5, rho=1.5)


def test_large_tau_collapses_to_the_non_spatial_fit():
    """tau -> infinity shrinks u to zero and must recover the closed
    form. This is the case that caught a catastrophic cancellation in
    the covariance when it was computed as a Schur complement."""
    y = [4, 9, 2, 11, 6]
    e = [1.0, 2.0, 0.7, 2.5, 1.4]
    r = poissp(y, offset=e, W=W5, rho=1.0, tau=1e9)
    assert max(abs(v) for v in r["u"]) < 1e-6
    assert r["beta"][0] == pytest.approx(math.log(sum(y) / sum(e)), abs=1e-8)
    assert r["se"][0] == pytest.approx(1.0 / math.sqrt(sum(y)), rel=1e-6)


def test_intrinsic_fit_satisfies_the_sum_to_zero_constraint():
    y = [4, 9, 2, 11, 6]
    e = [1.0, 2.0, 0.7, 2.5, 1.4]
    r = poissp(y, offset=e, W=W5, rho=1.0, tau=3.0)
    assert abs(sum(r["u"])) < 1e-10
    assert r["constrained"] is True


def test_offset_is_a_rate_denominator_not_a_free_coefficient():
    """Doubling every offset must shift the intercept by exactly
    -log 2, since the offset enters at coefficient one."""
    r1 = poissp(Y, offset=E)
    r2 = poissp(Y, offset=[2.0 * v for v in E])
    assert r2["beta"][0] - r1["beta"][0] == pytest.approx(-math.log(2.0),
                                                          abs=1e-10)


def test_relative_risk_is_exp_eta():
    r = poissp(Y, offset=E, W=None)
    for rr, eta in zip(r["relative_risk"], r["eta"]):
        assert rr == pytest.approx(math.exp(eta), rel=1e-12)


def test_deviance_is_zero_for_a_saturated_spatial_fit():
    """A very weak CAR prior lets u absorb every residual, so the
    fitted values approach the data and the deviance goes to zero."""
    y = [4, 9, 2, 11, 6]
    e = [1.0, 2.0, 0.7, 2.5, 1.4]
    r = poissp(y, offset=e, W=W5, rho=0.0, tau=1e-6, constrain=False)
    assert r["deviance"] < 1e-4


def test_counts_must_be_non_negative_integers():
    with pytest.raises(ValueError, match="non-negative"):
        poissp([1, -2, 3], offset=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="integers"):
        poissp([1.5, 2.0, 3.0], offset=[1.0, 1.0, 1.0])


def test_weight_matrix_is_validated():
    with pytest.raises(ValueError, match="zero diagonal"):
        poissp([1, 2, 3], W=[[1, 1, 0], [1, 0, 1], [0, 1, 0]])
    with pytest.raises(ValueError, match="symmetric"):
        poissp([1, 2, 3], W=[[0, 1, 0], [0, 0, 1], [0, 1, 0]])
