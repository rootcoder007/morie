"""Verification tests for kalmf.kalman_filter.

For a scalar random walk observed with noise the filter reduces to a
recursion that can be written out in four lines, so the filtered state,
its variance and the log-likelihood are all checked against it.
"""

import math

import pytest

from morie.fn.kalmf import kalman_filter


Y = [1.0, 0.7, 1.4, 0.2, 0.9]
Q = 0.1
R = 0.5


def _scalar_recursion(y, q, r, x0=0.0, p0=1.0):
    """The textbook scalar filter: predict, then correct."""
    x, p = x0, p0
    states, vars_, ll = [], [], 0.0
    for obs in y:
        # predict
        xp = x
        pp = p + q
        # the one-step-ahead forecast has variance pp + r
        v = obs - xp
        f = pp + r
        ll += -0.5 * (math.log(2.0 * math.pi * f) + v * v / f)
        # correct
        k = pp / f
        x = xp + k * v
        p = (1.0 - k) * pp
        states.append(x)
        vars_.append(p)
    return states, vars_, ll


def test_filtered_states_match_the_scalar_recursion():
    res = kalman_filter(Y, [[1.0]], [[1.0]], [[Q]], [[R]],
                        x0=[0.0], P0=[[1.0]])
    states, _, _ = _scalar_recursion(Y, Q, R)
    got = [float(s[0]) for s in res["state"]]
    for a, b in zip(got, states):
        assert a == pytest.approx(b, rel=1e-10)


def test_filtered_variances_match_the_scalar_recursion():
    res = kalman_filter(Y, [[1.0]], [[1.0]], [[Q]], [[R]],
                        x0=[0.0], P0=[[1.0]])
    _, vars_, _ = _scalar_recursion(Y, Q, R)
    got = [float(P[0][0]) for P in res["cov"]]
    for a, b in zip(got, vars_):
        assert a == pytest.approx(b, rel=1e-10)


def test_log_likelihood_matches_the_sum_of_forecast_densities():
    res = kalman_filter(Y, [[1.0]], [[1.0]], [[Q]], [[R]],
                        x0=[0.0], P0=[[1.0]])
    _, _, ll = _scalar_recursion(Y, Q, R)
    assert float(res["loglik"]) == pytest.approx(ll, rel=1e-9)


def test_trusting_the_data_makes_the_filter_follow_it():
    # with the process noise far larger than the measurement noise the
    # gain approaches one, so each filtered state sits on its observation
    res = kalman_filter(Y, [[1.0]], [[1.0]], [[1e6]], [[1e-9]],
                        x0=[0.0], P0=[[1.0]])
    got = [float(s[0]) for s in res["state"]]
    for a, b in zip(got, Y):
        assert a == pytest.approx(b, rel=1e-6)


def test_distrusting_the_data_holds_the_state_still():
    # the mirror case: no process noise at all pins the state, so it
    # cannot chase a varying series
    res = kalman_filter(Y, [[1.0]], [[1.0]], [[0.0]], [[1.0]],
                        x0=[0.0], P0=[[1e-9]])
    got = [float(s[0]) for s in res["state"]]
    assert max(abs(v) for v in got) < 1e-3


def test_a_non_square_transition_matrix_is_refused():
    with pytest.raises(ValueError):
        kalman_filter(Y, [[1.0, 0.0]], [[1.0]], [[Q]], [[R]])
