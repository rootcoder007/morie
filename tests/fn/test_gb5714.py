"""Tests for gb5714.gibbons_wsrt_sampsize."""

import pytest

from morie.fn.gb5714 import gibbons_wsrt_sampsize


def test_gb5714_basic():
    """Eq. (5.7.15): N = (z_a + z_b)^2 / (3 (p2 - 1/2)^2), recomputed here
    and against scipy.stats.norm (1150.3160944283197 for p2 = 0.556)."""
    import math
    r = gibbons_wsrt_sampsize(0.556, alpha=0.05, beta=0.05)
    z = 1.6448536269514722  # scipy.stats.norm.ppf(0.95)
    assert r["n_raw"] == pytest.approx((2 * z) ** 2 / (3 * 0.056 ** 2), rel=1e-12)
    assert r["n_raw"] == pytest.approx(1150.3160944283197, rel=1e-12)
    assert r["n"] == math.ceil(r["n_raw"]) == 1151
    assert gibbons_wsrt_sampsize(0.921)["n"] == 21


def test_gb5714_edge():
    """Two-sided halves alpha; p2 = 1/2 has no finite N."""
    r = gibbons_wsrt_sampsize(0.556, twosided=True)
    assert r["z_alpha"] == pytest.approx(1.959963984540054, rel=1e-12)
    assert r["n_raw"] == pytest.approx(1381.240434961676, rel=1e-12)
    with pytest.raises(ValueError, match="differ from 0.5"):
        gibbons_wsrt_sampsize(0.5)
    with pytest.raises(ValueError, match="inside"):
        gibbons_wsrt_sampsize(1.2)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gb5714 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
