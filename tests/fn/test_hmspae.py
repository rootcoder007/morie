"""Tests for hmspae.geron_sparse_autoencoder."""

import math

from morie.fn import _array_core as np

from morie.fn.hmspae import geron_sparse_autoencoder


def test_hmspae_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # geron_sparse_autoencoder expects activations (post-sigmoid) in [0, 1].
    z = rng.normal(0, 1, (40, 3))
    X = [[1.0 / (1.0 + math.exp(-x)) for x in row] for row in z]
    rho = 0.5
    beta = 0.8
    result = geron_sparse_autoencoder(X, rho, beta)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_hmspae_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # geron_sparse_autoencoder expects activations (post-sigmoid) in [0, 1].
    z = rng.normal(0, 1, (20, 2))
    X = [[1.0 / (1.0 + math.exp(-x)) for x in row] for row in z]
    rho = 0.1
    beta = 3.0
    result = geron_sparse_autoencoder(X, rho, beta)
    assert isinstance(result, dict)
    assert len(result) > 0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmspae as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
