"""Tests for hierdp.hierarchical_dp_density."""

import pytest

from morie.fn import _array_core as np
from morie.fn.hierdp import hierarchical_dp_density


def test_hierdp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    groups = [rng.normal(0, 1, 40), rng.normal(3, 1, 40), rng.normal(-2, 1, 40)]
    result = hierarchical_dp_density(groups, n_iter=60, seed=0)
    assert isinstance(result, dict)
    assert "densities" in result
    assert "at" in result
    assert "n_components" in result
    assert "sharing_index" in result
    assert "components_per_group" in result
    assert "atoms" in result
    assert result["densities"].shape[0] == 3
    assert len(result["components_per_group"]) == 3


def test_hierdp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    # Single group is invalid (need at least 2 groups)
    with pytest.raises(ValueError):
        hierarchical_dp_density([rng.normal(0, 1, 10)])
    # Group with fewer than 2 observations is invalid
    with pytest.raises(ValueError):
        hierarchical_dp_density([rng.normal(0, 1, 1), rng.normal(0, 1, 10)])
    # Non-positive alpha is invalid
    with pytest.raises(ValueError):
        hierarchical_dp_density(
            [rng.normal(0, 1, 10), rng.normal(0, 1, 10)], alpha=0.0
        )
    # Non-positive gamma is invalid
    with pytest.raises(ValueError):
        hierarchical_dp_density(
            [rng.normal(0, 1, 10), rng.normal(0, 1, 10)], gamma=-1.0
        )


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hierdp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
