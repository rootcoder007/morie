"""survnnr: DeepSurv -- Cox partial likelihood by gradient descent.

The generated test imported `survival_neural_net`, which does not exist.
Rewritten against deep_surv. With hidden=() the model IS Cox regression,
which gives a checkable anchor: the sign of the recovered coefficient.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.survnnr import deep_surv, concordance


def _risky_cohort(n=60):
    """Higher x means higher risk, so shorter survival."""
    X = [[float(i % 10) / 10.0] for i in range(n)]
    times = [10.0 - float(i % 10) * 0.8 for i in range(n)]
    events = [1] * n
    return X, times, events


def test_recovers_a_positive_coefficient_when_risk_rises_with_x():
    """Shorter times for larger x is positive association; a Cox fit must
    return a positive coefficient. A sign error would fail here."""
    X, t, e = _risky_cohort()
    r = deep_surv(X, t, e, hidden=(), n_epochs=300, seed=0)
    assert float(np.asarray(r["coefficients"])[0]) > 0


def test_loss_decreases_over_training():
    X, t, e = _risky_cohort()
    r = deep_surv(X, t, e, hidden=(), n_epochs=200, seed=0)
    hist = [float(v) for v in np.asarray(r["loss_history"])]
    assert hist[-1] <= hist[0]


def test_concordance_beats_chance_on_data_with_signal():
    X, t, e = _risky_cohort()
    r = deep_surv(X, t, e, hidden=(), n_epochs=300, seed=0)
    c = float(concordance(r, X, t, e)["c_index"])
    assert c > 0.5
