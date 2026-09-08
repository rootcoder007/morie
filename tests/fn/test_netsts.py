"""Tests for netsts. Full anchor: wave3/anchor_intermittent.py."""
import pytest
from morie.fn import _s03core as k
from morie.fn.netsts import (gradient_retention, lstm_forecast,
                             standardize)


def test_retention_is_f_to_the_t_exactly():
    """The constant error carousel, as a number: the cell path is
    multiplication by f alone."""
    assert gradient_retention(0.9, 10) == pytest.approx(0.9 ** 10,
                                                        abs=1e-15)
    assert gradient_retention(1.0, 10000) == 1.0
    assert gradient_retention(0.99, 200) > 0.1
    assert gradient_retention(0.5, 200) < 1e-50
    with pytest.raises(ValueError):
        gradient_retention(1.5, 10)


def test_a_zero_forget_bias_destroys_memory_in_a_dozen_steps():
    assert k.sigmoid(0.0) == pytest.approx(0.5)
    assert gradient_retention(k.sigmoid(0.0), 12) < 0.001
    assert gradient_retention(k.sigmoid(2.0), 12) > 0.1


def test_standardisation_keeps_tanh_off_its_rails():
    zs, mu, sd = standardize([100.0, 200.0, 300.0])
    assert k.mean(zs) == pytest.approx(0.0, abs=1e-12)
    assert max(abs(v) for v in zs) < 2.0


def test_recursive_and_direct_strategies():
    lin = [float(t) for t in range(60)]
    rec = lstm_forecast(lin, 6, hidden=4, n_lags=4,
                        strategy="recursive", seed=1)
    dirf = lstm_forecast(lin, 6, hidden=4, n_lags=4, strategy="direct",
                         seed=1)
    assert rec["n_models"] == 1
    assert dirf["n_models"] == 6
    assert rec["forecast"][-1] > rec["forecast"][0]
    assert dirf["forecast"][-1] > dirf["forecast"][0]
    with pytest.raises(ValueError):
        lstm_forecast(lin, 6, strategy="nope")
    with pytest.raises(ValueError):
        lstm_forecast(lin[:5], 6)
