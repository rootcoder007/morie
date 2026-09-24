"""Tests for irt1pl.rasch_one_parameter."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.irt1pl import rasch_one_parameter


def test_irt1pl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(0)
    n = 40
    y = rng.integers(0, 2, size=n)  # binary responses
    theta = rng.normal(0, 1, size=n)  # person abilities
    b = 0.5  # item difficulty
    result = rasch_one_parameter(y, theta, b)
    # result should be a dict-like object (RichResult)
    assert isinstance(result, dict)
    # check that the documented payload keys are present
    for key in ("estimate", "p", "information", "loglik", "b", "n", "method"):
        assert key in result
    # shapes of returned arrays
    assert len(result["p"]) == n
    assert len(result["information"]) == n
    # numeric properties of scalar outputs
    est = result["estimate"]
    assert 0.0 <= est <= 1.0
    assert math.isfinite(result["loglik"])
    # stored parameters
    assert result["b"] == b
    assert result["n"] == n
    # method description is a string
    assert isinstance(result["method"], str)


def test_irt1pl_edge():
    """Test edge cases."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=5)
    theta = rng.normal(0, 1, size=4)  # length mismatch should raise
    b = 0.3
    with pytest.raises(ValueError):
        rasch_one_parameter(y, theta, b)
