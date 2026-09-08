"""Tests for shewh.shewhart (Shewhart 1926 control chart).

The chart returns a decision, so the checks are a confusion matrix on
fixtures where the in-control/out-of-control truth is known by
construction, including both degenerate cases.
"""

import pytest

from morie.fn.shewh import shewhart

IC = [0.0, 1.2, -2.1, 0.4, 2.9, -1.7, 0.0, 2.99]
OC = [6.0, -9.5, 3.01]


def confusion(x, truth, mu=0.0, sigma=1.0, k=3.0):
    a = shewhart(x, mu, sigma, k)["alerts"]
    tp = sum(1 for i in range(len(x)) if truth[i] == 1 and a[i] == 1)
    fp = sum(1 for i in range(len(x)) if truth[i] == 0 and a[i] == 1)
    tn = sum(1 for i in range(len(x)) if truth[i] == 0 and a[i] == 0)
    fn = sum(1 for i in range(len(x)) if truth[i] == 1 and a[i] == 0)
    return tp, fp, tn, fn


def test_confusion_matrix_separable_fixture():
    x = IC + OC
    truth = [0] * len(IC) + [1] * len(OC)
    assert confusion(x, truth) == (3, 0, 8, 0)


def test_confusion_matrix_wrong_both_ways():
    """A genuine 3.5-sigma tail draw from the in-control process is a false
    alarm; a shifted draw landing at 2.5 sigma is a miss."""
    in_control = [0.2, -0.9, 3.5, 1.1, -2.4, 0.0]
    shifted = [7.1, 4.6, 2.5, 5.2]
    x = in_control + shifted
    truth = [0] * len(in_control) + [1] * len(shifted)
    assert confusion(x, truth) == (3, 1, 5, 1)


def test_degenerate_cases():
    assert shewhart(IC, 0.0, 1.0)["n_alerts"] == 0
    assert shewhart(OC, 0.0, 1.0)["n_alerts"] == len(OC)


def test_false_alarm_rate_of_the_three_sigma_limit():
    r = shewhart([0.0, 1.0], 0.0, 1.0, 3.0)
    assert abs(r["false_alarm_prob"] - 0.0026997960632601883) < 1e-14
    assert abs(r["arl0"] - 370.398) < 0.01


def test_limits_and_boundary():
    r = shewhart([0.0], 7.0, 2.0, 3.0)
    assert r["ucl"] == 13.0 and r["lcl"] == 1.0
    assert shewhart([3.0], 0.0, 1.0, 3.0)["n_alerts"] == 0
    assert shewhart([3.0 + 1e-12], 0.0, 1.0, 3.0)["n_alerts"] == 1


def test_tighter_limits_flag_at_least_as_much():
    x = IC + OC
    assert (shewhart(x, 0.0, 1.0, 2.0)["n_alerts"]
            >= shewhart(x, 0.0, 1.0, 3.0)["n_alerts"])
    assert (shewhart(x, 0.0, 1.0, 2.0)["false_alarm_prob"]
            > shewhart(x, 0.0, 1.0, 3.0)["false_alarm_prob"])


def test_error_paths():
    with pytest.raises(ValueError):
        shewhart([], 0.0, 1.0)
    with pytest.raises(ValueError):
        shewhart([1.0], 0.0, 0.0)
    with pytest.raises(ValueError):
        shewhart([1.0], 0.0, -2.0)
    with pytest.raises(ValueError):
        shewhart([1.0], 0.0, 1.0, 0.0)
