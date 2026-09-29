"""Tests for grpio.geron_perceiver_io."""

import doctest as _doctest

import morie.fn.grpio as _doctest_module
from morie.fn.grpio import geron_perceiver_io


def test_grpio_basic():
    """Test basic functionality."""
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    Z_latent = [[0.0, 0.0]]
    output_queries = [[0.0, 0.0]]
    result = geron_perceiver_io(X, Z_latent, output_queries)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grpio_edge():
    """Test edge cases."""
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    Z_latent = [[0.0, 0.0]]
    output_queries = [[0.0, 0.0]]
    result = geron_perceiver_io(X, Z_latent, output_queries)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False, optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS
    )
    assert res.attempted > 0
    assert res.failed == 0


def test_perceiver_cross_attention_recomputed():
    """One latent attending to the inputs: softmax(z X' / sqrt d) X."""
    import math

    import pytest

    X = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [0.5, -1.0]]
    z0 = [0.3, 0.2]
    s = [sum(a * b for a, b in zip(z0, x)) / math.sqrt(2) for x in X]
    e = [math.exp(v - max(s)) for v in s]
    w = [v / sum(e) for v in e]
    z1 = [sum(w[i] * X[i][c] for i in range(4)) for c in range(2)]
    r = geron_perceiver_io(X, [z0], [[1.0, 0.0]])
    assert r["cross_weights"][0][0] == pytest.approx(w, rel=1e-13)
    # self-attention of a single latent returns it unchanged
    assert r["latent"][0] == pytest.approx(z1, rel=1e-13)
    assert r["output"][0] == pytest.approx(z1, rel=1e-13)
