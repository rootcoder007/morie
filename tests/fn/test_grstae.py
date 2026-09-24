"""Tests for grstae.geron_stacked_autoencoder."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.grstae import geron_stacked_autoencoder


def test_grstae_basic():
    """Test basic functionality."""
    W = [[1.0], [1.0]]
    x = [[1.0, 2.0]]
    result = geron_stacked_autoencoder(x, [W], activation="linear")

    # Check that the result is a dict-like RichResult
    assert isinstance(result, dict)

    # Check that the documented keys are present
    expected_keys = {
        "reconstruction",
        "code",
        "activations",
        "reconstruction_error",
        "compression",
        "estimate",
        "n",
        "method",
    }
    assert expected_keys.issubset(result.keys())

    # Validate code shape and value
    code = np.array(result["code"])
    assert code.shape == (1, 1)
    assert math.isclose(float(code[0, 0]), 3.0)

    # Validate reconstruction shape and values
    recon = np.array(result["reconstruction"])
    assert recon.shape == (1, 2)
    assert math.isclose(float(recon[0, 0]), 3.0)
    assert math.isclose(float(recon[0, 1]), 3.0)

    # Compression ratio and reconstruction error as per the docstring example
    assert math.isclose(float(result["compression"]), 2.0)
    assert math.isclose(float(result["reconstruction_error"]), 2.5)

    # n should be a positive integer (number of samples)
    assert isinstance(result["n"], int)
    assert result["n"] > 0

    # method is the human-readable description, not the function name
    assert result["method"] == "Stacked (deep) autoencoder forward pass"


def test_grstae_edge():
    """Test edge cases."""
    # Empty input raises ValueError
    with pytest.raises(ValueError):
        geron_stacked_autoencoder([], [[1.0]])

    # Empty layer_weights raises ValueError
    with pytest.raises(ValueError):
        geron_stacked_autoencoder([[1.0, 2.0]], [])

    # No bottleneck (code dimension not narrower than input) raises ValueError
    W_identity = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        geron_stacked_autoencoder([[1.0, 2.0]], [W_identity], activation="linear")


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grstae as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
