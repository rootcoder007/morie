"""Tests for hmddim.geron_ddim."""

from morie.fn import _array_core as np

from morie.fn.hmddim import geron_ddim


def test_hmddim_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x_T = rng.normal(0, 1, 5)

    def zero_model(x, t):
        return np.zeros(len(x))

    result = geron_ddim(
        x_T, zero_model, T=4, n_steps=2,
        beta_schedule=[0.5, 0.5, 0.5, 0.5],
    )
    assert isinstance(result, dict)
    assert "x_0" in result
    assert "trajectory" in result
    assert "timesteps" in result
    assert "model_calls" in result
    assert "speedup" in result
    assert result["model_calls"] == 2
    assert result["timesteps"] == [4, 1]
    assert result["speedup"] == 2.0


def test_hmddim_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x_T = rng.normal(0, 1, 3)

    def zero_model(x, t):
        return np.zeros(len(x))

    # smallest valid configuration: T=1, n_steps=1
    result = geron_ddim(
        x_T, zero_model, T=1, n_steps=1,
        beta_schedule=[0.5],
    )
    assert isinstance(result, dict)
    assert "x_0" in result
    assert "trajectory" in result
    assert "timesteps" in result
    assert "model_calls" in result
    assert "speedup" in result
    assert result["model_calls"] == 1
    assert result["speedup"] == 1.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmddim as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
