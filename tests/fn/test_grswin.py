"""Tests for grswin.geron_swin_window_attention."""

from morie.fn import _array_core as np

from morie.fn.grswin import geron_swin_window_attention


def test_grswin_basic():
    """Test basic functionality with a non-trivial 3-D feature map."""
    rng = np.random.default_rng(42)
    H, W, d_model = 8, 8, 4
    d_k, d_v = 4, 6
    X = rng.normal(0, 1, (H, W, d_model))
    window_size = 4
    WQ = rng.normal(0, 1, (d_model, d_k))
    WK = rng.normal(0, 1, (d_model, d_k))
    WV = rng.normal(0, 1, (d_model, d_v))
    result = geron_swin_window_attention(X, window_size, WQ, WK, WV)
    assert isinstance(result, dict)
    assert result["n_windows"] == (H // window_size) * (W // window_size)
    assert result["tokens_per_window"] == window_size * window_size
    out = np.asarray(result["output"])
    assert out.shape == (H, W, d_v)


def test_grswin_edge():
    """Test edge case following the docstring example: 2x2 map, window_size 1."""
    X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    I = [[1.0]]
    result = geron_swin_window_attention(X, 1, I, I, I)
    assert isinstance(result, dict)
    assert result["n_windows"] == 4
    assert result["tokens_per_window"] == 1
    out = np.asarray(result["output"])
    assert out.shape == (2, 2, 1)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grswin as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
