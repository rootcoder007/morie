"""Tests for hmmlpf.geron_mlp."""

from morie.fn import _array_core as np

from morie.fn.hmmlpf import geron_mlp


def test_hmmlpf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_in = 5
    hidden = 4
    n_out = 3
    m = 40

    X = rng.normal(0, 1, (m, n_in))
    W0 = rng.normal(0, 1, (n_in, hidden))
    W1 = rng.normal(0, 1, (hidden, n_out))
    weights = [W0, W1]

    biases = [np.zeros(hidden), np.zeros(n_out)]
    activations = ["relu", "softmax"]

    result = geron_mlp(X, weights, biases, activations)
    assert isinstance(result, dict)
    assert "output" in result
    assert "n_parameters" in result
    assert "pre_activations" in result
    assert "activations" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    output = result["output"]
    assert len(output) == m
    for row in output:
        assert len(row) == n_out

    expected_n_params = n_in * hidden + hidden + hidden * n_out + n_out
    assert result["n_parameters"] == expected_n_params

    assert result["n"] == m


def test_hmmlpf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_in = 3
    n_out = 2

    # Single-sample 1-D input is accepted by the docstring.
    X = rng.normal(0, 1, n_in)
    weights = [rng.normal(0, 1, (n_in, n_out))]
    biases = [np.zeros(n_out)]
    activations = ["tanh"]

    result = geron_mlp(X, weights, biases, activations)
    assert isinstance(result, dict)
    assert "output" in result

    output = result["output"]
    assert len(output) == 1
    assert len(output[0]) == n_out

    expected_n_params = n_in * n_out + n_out
    assert result["n_parameters"] == expected_n_params


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmmlpf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
