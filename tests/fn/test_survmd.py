"""Tests for survmd.survival_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.survmd import survival_mediation


def test_survmd_basic():
    rng = np.random.default_rng(42)
    n = 4000
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(scale=0.5, size=n)
    t_event = rng.exponential(np.exp(-(0.4 * x + 0.6 * m)))
    cens = rng.exponential(2.0, size=n)
    time = np.minimum(t_event, cens)
    event = (t_event <= cens).astype(float)
    out = survival_mediation(time, event, x, m)
    assert out["coefficients"]["theta1"] == pytest.approx(0.4, abs=0.1)
    assert out["hr_total"] == pytest.approx(out["hr_nde"] * out["hr_nie"])


def test_survmd_edge():
    with pytest.raises(ValueError):
        survival_mediation([1.0] * 20, np.zeros(20), np.zeros(20), np.zeros(20))  # no events
    with pytest.raises(ValueError):
        survival_mediation([-1.0] * 20, np.ones(20), np.zeros(20), np.zeros(20))
