"""Tests for bkmed.baron_kenny."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bkmed import baron_kenny


def _mediated(seed=0, n=600, a=0.8, b=0.6, direct=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    m = a * x + rng.normal(0, 1, n)
    y = direct * x + b * m + rng.normal(0, 1, n)
    return y, x, m


def test_recovers_the_generating_paths():
    y, x, m = _mediated(seed=1, a=0.8, b=0.6, direct=0.3)
    r = baron_kenny(y, x, m)
    assert r["a"] == pytest.approx(0.8, abs=0.12)
    assert r["b"] == pytest.approx(0.6, abs=0.12)
    assert r["c_prime"] == pytest.approx(0.3, abs=0.15)


def test_total_effect_decomposes_into_direct_plus_indirect():
    """c = c' + ab holds exactly in OLS with these three regressions."""
    y, x, m = _mediated(seed=2, direct=0.4)
    r = baron_kenny(y, x, m)
    assert r["c"] == pytest.approx(r["c_prime"] + r["a"] * r["b"], rel=1e-10)


def test_complete_mediation_is_called_when_there_is_no_direct_path():
    y, x, m = _mediated(seed=3, direct=0.0)
    r = baron_kenny(y, x, m)
    assert r["mediation"] == "complete"
    assert r["p"]["c_prime"] > 0.05


def test_partial_mediation_is_called_when_a_direct_path_survives():
    y, x, m = _mediated(seed=4, direct=0.6)
    r = baron_kenny(y, x, m)
    assert r["mediation"] == "partial"


def test_no_mediation_when_the_mediator_is_noise():
    rng = np.random.default_rng(5)
    n = 600
    x = rng.normal(0, 1, n)
    m = rng.normal(0, 1, n)          # unrelated to x
    y = 0.7 * x + rng.normal(0, 1, n)
    assert baron_kenny(y, x, m)["mediation"] == "none"


def test_steps_are_reported_separately():
    """So a caller can apply a weaker rule than requiring step 1."""
    y, x, m = _mediated(seed=6, direct=0.0)
    steps = baron_kenny(y, x, m)["steps"]
    assert set(steps) == {
        "step1_total_effect_significant",
        "step2_x_predicts_m",
        "step3_m_predicts_y_given_x",
        "step4_direct_effect_shrinks",
    }
    assert steps["step2_x_predicts_m"] and steps["step3_m_predicts_y_given_x"]


def test_validates_inputs():
    y, x, m = _mediated(seed=7, n=50)
    with pytest.raises(ValueError, match="one-dimensional"):
        baron_kenny(y, np.column_stack([x, x]), m)
    with pytest.raises(ValueError, match="same length"):
        baron_kenny(y, x, m[:-1])
    with pytest.raises(ValueError, match="at least 4 observations"):
        baron_kenny(y[:3], x[:3], m[:3])
    with pytest.raises(ValueError, match="must be finite"):
        bad = y.copy(); bad[0] = np.inf
        baron_kenny(bad, x, m)
