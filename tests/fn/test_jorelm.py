"""Tests for jorelm.joseph_relative_mae."""

from morie.fn import _array_core as np
import math

from morie.fn.jorelm import joseph_relative_mae


def test_jorelm_basic():
    """Test basic functionality."""
    rng_true = np.random.default_rng(43)
    y_true = rng_true.normal(0, 1, 100)
    rng_pred = np.random.default_rng(44)
    y_pred = rng_pred.normal(0, 1, 100)
    rng_base = np.random.default_rng(42)
    y_baseline = rng_base.normal(0, 1, 100)
    result = joseph_relative_mae(y_true, y_pred, y_baseline)
    assert hasattr(result, "relmae")
    assert hasattr(result, "mae")
    assert hasattr(result, "benchmae")
    assert math.isfinite(result.relmae)
    assert math.isfinite(result.mae)
    assert math.isfinite(result.benchmae)


def test_jorelm_edge():
    """Test edge cases."""
    rng_true = np.random.default_rng(43)
    y_true = rng_true.normal(0, 1, 20)
    rng_pred = np.random.default_rng(44)
    y_pred = rng_pred.normal(0, 1, 20)
    rng_base = np.random.default_rng(42)
    y_baseline = rng_base.normal(0, 1, 20)
    result = joseph_relative_mae(y_true, y_pred, y_baseline)
    assert hasattr(result, "relmae")
    assert math.isfinite(result.relmae)
