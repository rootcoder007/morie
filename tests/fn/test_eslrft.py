"""Tests for eslrft.esl_random_forest."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.eslrft import esl_random_forest


# The implementation calls np.setdiff1d with an ``assume_unique`` keyword
# that the pure-Python _array_core stub does not accept. Patch it on the
# module so the source-level call succeeds.
_orig_setdiff1d = np.setdiff1d
def _patched_setdiff1d(ar1, ar2, assume_unique=False):
    return _orig_setdiff1d(ar1, ar2)
np.setdiff1d = _patched_setdiff1d


def test_eslrft_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (40, 3))
    y = rng_y.normal(0, 1, 40)
    result = esl_random_forest(X, y, B=5, seed=0)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "oob_prediction" in result
    assert "oob_mse" in result
    assert "train_mse" in result
    assert "mtry" in result
    assert "mtry_rule" in result
    assert "B" in result
    assert "n" in result
    assert "p" in result
    assert "n_oob_missing" in result
    assert "subset_drawn_per" in result
    assert "method" in result
    assert result["B"] == 5
    assert result["n"] == 40
    assert result["p"] == 3
    assert math.isfinite(result["train_mse"])
    assert math.isfinite(result["oob_mse"])
    pred = np.asarray(result["prediction"])
    assert pred.shape == (40,)


def test_eslrft_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (40, 3))
    y = rng_y.normal(0, 1, 40)
    result = esl_random_forest(X, y, B=2, mtry=1, max_depth=3,
                               min_node=2, seed=1)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert result["mtry"] == 1
    assert result["B"] == 2
    assert result["n"] == 40
    assert result["p"] == 3
