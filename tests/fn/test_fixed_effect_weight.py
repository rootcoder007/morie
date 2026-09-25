"""Tests for fixed_effect_weight.fixed_effect_weight."""

import pytest

from morie.fn.fixed_effect_weight import fixed_effect_weight


def test_fixed_effect_weight_basic():
    """Eq. (11.34): w = 1 / se^2."""
    assert fixed_effect_weight(0.25)["value"] == 16.0
    assert fixed_effect_weight(0.3)["value"] == pytest.approx(1 / 0.09, rel=1e-15)


def test_fixed_effect_weight_edge():
    with pytest.raises(ValueError, match="positive"):
        fixed_effect_weight(0.0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import importlib as _importlib

# the package also exports a function of this name, so the import
# statement would bind that function, not the module
_doctest_module = _importlib.import_module("morie.fn.fixed_effect_weight")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
