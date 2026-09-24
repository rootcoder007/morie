"""Tests for hmonnx.geron_onnx_export."""

import pytest

from morie.fn import _array_core as np

from morie.fn.hmonnx import geron_onnx_export


def test_hmonnx_basic():
    """Test basic functionality."""
    model = [
        {"op": "Gemm", "in_features": 3, "out_features": 2},
        {"op": "Relu"},
    ]
    args = np.zeros((1, 3))
    result = geron_onnx_export(model, args)
    assert isinstance(result, dict)
    assert result["input_shape"] == (1, 3)
    assert result["output_shape"] == (1, 2)
    assert len(result["nodes"]) == 2
    assert result["is_protobuf"] is False


def test_hmonnx_edge():
    """Test edge cases: shape mismatch raises ValueError."""
    model = [
        {"op": "Gemm", "in_features": 3, "out_features": 2},
        {"op": "Relu"},
    ]
    args = np.zeros((1, 5))
    with pytest.raises(ValueError):
        geron_onnx_export(model, args)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmonnx as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
