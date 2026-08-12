"""Tests for ukfF (unscented Kalman filter).

Replaces the generated stub, which imported ``unscented_kalman``.
"""

from morie.fn.ukfF import ukfF


def test_a_linear_system_matches_the_ordinary_kalman_filter():
    # scalar random walk observed directly: the UKF must reproduce the
    # exact Kalman recursion, since the transforms are linear
    q, r = 0.1, 0.5
    zs = [[1.0], [1.2], [0.9], [1.5], [1.1]]   # one vector per step
    res = ukfF(lambda x: x, lambda x: x, [[q]], [[r]], [0.0], [[1.0]], zs)
    # the textbook scalar filter, by hand
    x, p = 0.0, 1.0
    for zv in zs:
        z = zv[0]
        x_pred, p_pred = x, p + q
        k = p_pred / (p_pred + r)
        x = x_pred + k * (z - x_pred)
        p = (1.0 - k) * p_pred
    assert abs(res["states"][-1][0] - x) < 1e-8
    assert abs(res["covariances"][-1][0][0] - p) < 1e-8


def test_the_filter_tracks_a_moving_target():
    zs = [[float(i)] for i in range(20)]
    res = ukfF(lambda x: x, lambda x: x, [[1.0]], [[0.1]], [0.0],
               [[1.0]], zs)
    assert abs(res["states"][-1][0] - 19.0) < 1.0


def test_more_measurement_noise_means_less_trust_in_the_data():
    zs = [[0.0], [5.0], [0.0], [5.0], [0.0]]
    trusting = ukfF(lambda x: x, lambda x: x, [[0.1]], [[0.01]], [0.0],
                    [[1.0]], zs)["states"]
    sceptical = ukfF(lambda x: x, lambda x: x, [[0.1]], [[100.0]], [0.0],
                     [[1.0]], zs)["states"]
    swing_t = max(s[0] for s in trusting) - min(s[0] for s in trusting)
    swing_s = max(s[0] for s in sceptical) - min(s[0] for s in sceptical)
    assert swing_s < swing_t


def test_one_state_per_measurement_and_innovations_are_reported():
    zs = [[1.0], [2.0], [3.0]]
    res = ukfF(lambda x: x, lambda x: x, [[0.1]], [[0.1]], [0.0],
               [[1.0]], zs)
    assert len(res["states"]) == 3
    assert len(res["covariances"]) == 3
    assert len(res["innovations"]) == 3


def test_a_nonlinear_measurement_still_converges():
    # observe the square of a constant state
    truth = 2.0
    zs = [[truth ** 2]] * 15
    res = ukfF(lambda x: x, lambda x: [x[0] ** 2], [[1e-4]], [[1e-2]],
               [1.5], [[1.0]], zs)
    assert abs(res["states"][-1][0] - truth) < 0.2


def test_covariances_stay_positive():
    zs = [[1.0]] * 10
    res = ukfF(lambda x: x, lambda x: x, [[0.1]], [[0.1]], [0.0],
               [[1.0]], zs)
    assert all(c[0][0] > 0 for c in res["covariances"])
