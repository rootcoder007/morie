"""Tests for hmdrv.geron_diffusion_reverse."""

from morie.fn import _array_core as np

from morie.fn.hmdrv import geron_diffusion_reverse


def test_hmdrv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x_T = rng.normal(0, 1, 5)
    model = lambda x, t: np.zeros_like(x)
    T = 5
    result = geron_diffusion_reverse(x_T, model, T, beta_schedule="linear", seed=0)
    assert isinstance(result, dict)
    assert "x_0" in result
    assert "trajectory" in result
    assert "means" in result
    assert "betas" in result
    assert "alpha_bar" in result
    assert "n_steps" in result
    assert "model_calls" in result
    assert result["n_steps"] == T
    assert result["model_calls"] == T


def test_hmdrv_edge():
    """Test edge cases."""
    x_T = [1.0]
    model = lambda x, t: np.zeros_like(x)
    T = 1
    result = geron_diffusion_reverse(x_T, model, T, beta_schedule=[0.75])
    assert isinstance(result, dict)
    assert "x_0" in result
    assert "model_calls" in result
    assert result["model_calls"] == 1
    assert len(result["x_0"]) == 1


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmdrv as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
