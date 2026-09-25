"""Tests for gb_ssj.gibbons_sign_sample_size_2."""

import math

import pytest

from morie.fn.gb_ssj import gibbons_sign_sample_size_2


def test_gb_ssj_basic():
    """Eq. (5.4.9) with alpha/2: N = [(sqrt(theta(1-theta)) z_beta +
    0.5 z_{alpha/2}) / (0.5 - theta)]^2; theta = 0.75 gives 37.7, so 38."""
    r = gibbons_sign_sample_size_2(0.75, alpha=0.05, beta=0.10)
    za, zb = 1.959963984540054, 1.2815515655446004
    nraw = ((math.sqrt(0.1875) * zb + 0.5 * za) / (0.5 - 0.75)) ** 2
    assert r["n_raw"] == pytest.approx(nraw, rel=1e-12)
    assert r["n"] == math.ceil(nraw) == 38


def test_gb_ssj_edge():
    """Symmetric in theta about 1/2; theta = 1/2 has no finite N."""
    assert gibbons_sign_sample_size_2(0.25)["n_raw"] == pytest.approx(
        gibbons_sign_sample_size_2(0.75)["n_raw"], rel=1e-14)
    with pytest.raises(ValueError, match="differ"):
        gibbons_sign_sample_size_2(0.5)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gb_ssj as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
