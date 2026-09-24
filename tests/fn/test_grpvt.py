"""Tests for grpvt.geron_pyramid_vit_stage."""

from morie.fn import _array_core as np

from morie.fn.grpvt import geron_pyramid_vit_stage


def test_grpvt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    H, W, d_model = 8, 8, 4
    d_k, d_v = 4, 4
    X = rng.normal(0, 1, (H, W, d_model))
    WQ = rng.normal(0, 1, (d_model, d_k))
    WK = rng.normal(0, 1, (d_model, d_k))
    WV = rng.normal(0, 1, (d_model, d_v))
    result = geron_pyramid_vit_stage(X, WQ, WK, WV, reduction_ratio=2)
    assert isinstance(result, dict)
    assert "output" in result
    assert "weights" in result
    assert "reduced_tokens" in result
    # output should have shape (H*W, d_v) = (64, 4)
    assert len(result["output"]) == H * W
    assert len(result["output"][0]) == d_v
    # with R=2 on 8x8, reduced_tokens = (H/R)*(W/R) = 16
    assert result["reduced_tokens"] == 16


def test_grpvt_edge():
    """Test edge case R=1 (plain self-attention, no reduction)."""
    X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    I = [[1.0]]
    result = geron_pyramid_vit_stage(X, I, I, I, reduction_ratio=1)
    assert isinstance(result, dict)
    assert result["reduced_tokens"] == 4


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grpvt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
