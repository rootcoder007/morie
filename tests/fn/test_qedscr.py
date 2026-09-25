"""Tests for qedscr.qed_drug_likeness."""

import math

from morie.fn import _array_core as np
from morie.fn.qedscr import qed_drug_likeness


def test_qedscr_basic():
    """Test basic functionality."""
    properties = {
        "MW": 46.07,
        "ALOGP": -0.31,
        "HBA": 1,
        "HBD": 1,
        "PSA": 20.23,
        "ROTB": 0,
        "AROM": 0,
        "ALERTS": 0,
    }
    result = qed_drug_likeness(properties)
    assert math.isfinite(result)
    assert 0.0 <= result <= 1.0


def test_qedscr_edge():
    """Test edge cases."""
    properties = {
        "MW": 78.11,
        "ALOGP": 2.13,
        "HBA": 0,
        "HBD": 0,
        "PSA": 0.0,
        "ROTB": 0,
        "AROM": 1,
        "ALERTS": 0,
    }
    result = qed_drug_likeness(properties)
    assert math.isfinite(result)
    assert 0.0 <= result <= 1.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.qedscr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
