"""Tests for hmtsc.geron_torchscript."""
import doctest as _doctest
import math

import morie.fn.hmtsc as _doctest_module
from morie.fn.hmtsc import geron_torchscript


def test_hmtsc_basic():
    """Test basic functionality."""
    W = [[1.0, 0.0], [0.0, 1.0]]
    model = [("linear", W), ("relu",)]
    example_inputs = [[1.0, -1.0]]
    result = geron_torchscript(model, example_inputs)
    assert isinstance(result, dict)
    for key in ("graph", "output", "replay", "max_diff", "n_nodes",
                "shapes", "estimate", "n", "method"):
        assert key in result
    assert result["n_nodes"] == 2
    assert math.isfinite(float(result["max_diff"]))
    assert float(result["max_diff"]) == 0.0
    assert len(result["shapes"]) == 2


def test_hmtsc_edge():
    """Test edge cases."""
    W = [[1.0], [1.0]]
    b = [0.5]
    model = [("linear", W), ("bias", b), ("sigmoid",)]
    example_inputs = [[1.0, -1.0]]
    result = geron_torchscript(model, example_inputs)
    assert isinstance(result, dict)
    assert result["n_nodes"] == 3
    assert math.isfinite(float(result["max_diff"]))
    assert math.isfinite(float(result["output"][0][0]))
    out_val = float(result["output"][0][0])
    assert 0.0 <= out_val <= 1.0


# --- appended: the module's own worked example as a gate -----------
def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
