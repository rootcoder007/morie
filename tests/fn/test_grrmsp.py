"""Tests for grrmsp.geron_rmsprop_update."""

import doctest as _doctest
import math

import pytest

from morie.fn import _array_core as np
from morie.fn.grrmsp import geron_rmsprop_update


def test_grrmsp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    params = rng.normal(0, 1, 5)
    grads = rng.normal(0, 1, 5)
    result = geron_rmsprop_update(params, grads, lr=0.1, rho=0.9)
    assert isinstance(result, dict)
    for key in ("params", "state", "step_size", "effective_lr", "estimate"):
        assert key in result
    assert len(result["params"]) == 5
    assert len(result["step_size"]) == 5
    assert len(result["effective_lr"]) == 5
    for v in result["effective_lr"]:
        assert math.isfinite(float(v))
        assert float(v) > 0


def test_grrmsp_edge():
    """Test edge cases."""
    # documented invalid inputs
    with pytest.raises(ValueError):
        geron_rmsprop_update([1.0], [1.0, 2.0])  # shape mismatch
    with pytest.raises(ValueError):
        geron_rmsprop_update([1.0], [1.0], rho=1.0)  # rho must be in [0, 1)
    with pytest.raises(ValueError):
        geron_rmsprop_update([1.0], [1.0], lr=0.0)  # lr must be positive
    # state carried between calls
    out1 = geron_rmsprop_update([1.0], [1.0], lr=0.1)
    out2 = geron_rmsprop_update(
        out1["params"], [1.0], state=out1["state"], lr=0.1
    )
    assert isinstance(out2, dict)
    assert "params" in out2
    assert len(out2["params"]) == 1


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import morie.fn.grrmsp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
