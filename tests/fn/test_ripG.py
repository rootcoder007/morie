"""Tests for ripG.ripley_g_function."""

import pytest

from morie.fn import _array_core as np

from morie.fn.ripG import ripley_g_function


def test_ripG_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    # generate points inside a rectangular window
    points = rng.uniform(0.0, 10.0, size=(n, 2))
    window = [0.0, 10.0, 0.0, 10.0]
    # distances at which to evaluate G
    r_vals = np.linspace(0.1, 5.0, 10)

    result = ripley_g_function(points, window, r_vals)

    # the function returns a RichResult (dict-like) with the documented keys
    assert isinstance(result, dict)
    for key in ("r", "g", "g_border", "nn", "csr", "lambda_hat", "n"):
        assert key in result

    # the returned r array should have the same length as the input
    assert len(result["r"]) == len(r_vals)
    assert len(result["g"]) == len(r_vals)
    # nearest‑neighbour distances are stored per point
    assert len(result["nn"]) == n
    assert result["n"] == n


def test_ripG_edge():
    """Test edge cases: a single point should raise a ValueError."""
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 10.0, size=(1, 2))
    window = [0.0, 10.0, 0.0, 10.0]
    r_vals = [0.5]

    with pytest.raises(ValueError):
        ripley_g_function(points, window, r_vals)
