"""surxgb: accelerated failure time model by gradient boosting.

The generated test imported `xgb_survival`, which does not exist.
Rewritten against survival_xgboost, whose interval-censored inputs
(y_lower, y_upper) the generated version did not supply at all.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.surxgb import survival_xgboost, aft_loss


def _intervals(n=40):
    X = [[float(i % 6), float((i * 3) % 4)] for i in range(n)]
    lower = [1.0 + float(i % 6) for i in range(n)]
    upper = [l + 1.0 for l in lower]
    return X, lower, upper


def test_boosting_reduces_the_aft_loss():
    """Each round fits the gradient; the loss must not increase overall."""
    X, lo, hi = _intervals()
    r = survival_xgboost(X, lo, hi, n_rounds=20, eta=0.1)
    hist = [float(v) for v in np.asarray(r["loss_history"])]
    assert hist[-1] <= hist[0]


def test_it_builds_the_number_of_trees_requested():
    X, lo, hi = _intervals()
    r = survival_xgboost(X, lo, hi, n_rounds=7)
    assert len(r["trees"]) == 7
    assert r["n_rounds"] == 7


def test_predictions_are_finite_and_one_per_row():
    X, lo, hi = _intervals()
    r = survival_xgboost(X, lo, hi, n_rounds=5)
    pred = np.asarray(r["prediction"])
    assert len(pred) == len(X)
    assert all(np.isfinite(float(v)) for v in pred)


def test_the_loss_is_smallest_inside_the_censoring_interval():
    """An AFT prediction that lands within [lower, upper] must cost less
    than one far outside it."""
    inside = float(aft_loss(2.0, 3.0, np.log(2.5), 1.0, "normal"))
    outside = float(aft_loss(2.0, 3.0, np.log(50.0), 1.0, "normal"))
    assert inside < outside
