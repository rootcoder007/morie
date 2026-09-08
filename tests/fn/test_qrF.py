"""Tests for qrF (quantile / pinball loss).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.qrF import qrF


def test_the_pinball_loss_by_hand():
    y = [1.0, 2.0, 3.0]
    p = [1.5, 1.0, 4.0]
    theta = 0.7
    res = qrF(y, p, theta=theta)
    want = []
    for i in range(3):
        u = y[i] - p[i]
        want.append(theta * u if u >= 0 else (theta - 1.0) * u)
    assert abs(res["total"] - sum(want)) < 1e-12
    assert abs(res["estimate"] - sum(want) / 3.0) < 1e-12
    for i in range(3):
        assert abs(res["losses"][i] - want[i]) < 1e-12


def test_a_perfect_forecast_costs_nothing():
    y = [1.0, 2.0, 3.0]
    assert qrF(y, y, theta=0.3)["estimate"] == 0.0


def test_the_median_loss_is_half_the_absolute_error():
    y = [1.0, 5.0]
    p = [3.0, 1.0]
    res = qrF(y, p, theta=0.5)
    want = sum(abs(y[i] - p[i]) for i in range(2)) / 2.0 / 2.0
    assert abs(res["estimate"] - want) < 1e-12


def test_a_high_quantile_punishes_under_prediction():
    y, p_low, p_high = [10.0], [5.0], [15.0]
    under = qrF(y, p_low, theta=0.9)["estimate"]
    over = qrF(y, p_high, theta=0.9)["estimate"]
    assert under > over          # missing high is expensive at theta=0.9
    under_l = qrF(y, p_low, theta=0.1)["estimate"]
    over_l = qrF(y, p_high, theta=0.1)["estimate"]
    assert over_l > under_l


def test_the_loss_is_minimised_at_the_true_quantile():
    y = [float(i) for i in range(101)]        # quantiles are known
    at_median = qrF(y, [50.0] * 101, theta=0.5)["estimate"]
    off_median = qrF(y, [70.0] * 101, theta=0.5)["estimate"]
    assert at_median < off_median
    at_q90 = qrF(y, [90.0] * 101, theta=0.9)["estimate"]
    off_q90 = qrF(y, [50.0] * 101, theta=0.9)["estimate"]
    assert at_q90 < off_q90


def test_validation():
    for call in (lambda: qrF([], []),
                 lambda: qrF([1.0], [1.0, 2.0]),
                 lambda: qrF([1.0], [1.0], theta=0.0),
                 lambda: qrF([1.0], [1.0], theta=1.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
