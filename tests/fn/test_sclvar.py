"""Tests for sclvar.selection_coefficient."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.sclvar import selection_coefficient


def test_sclvar_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_loci, n_demes = 8, 4
    counts = rng.integers(0, 50, (n_loci, n_demes))
    n_total = [[50] * n_demes for _ in range(n_loci)]
    result = selection_coefficient(
        counts, n_total=n_total, generations=10, n_e=1000.0
    )
    expected_keys = ("fst", "fst_mean", "alpha_locus", "outlier",
                     "selection_type", "s", "drift_dominates", "q_value")
    for key in expected_keys:
        assert key in result
    # fst_mean is a single genome-wide scalar.
    assert math.isfinite(float(result["fst_mean"]))
    # Per-locus quantities must have one entry per locus.
    assert len(np.asarray(result["fst"])) == n_loci
    assert len(np.asarray(result["alpha_locus"])) == n_loci
    assert len(np.asarray(result["outlier"])) == n_loci
    assert len(np.asarray(result["selection_type"])) == n_loci
    assert len(np.asarray(result["s"])) == n_loci
    assert len(np.asarray(result["drift_dominates"])) == n_loci
    assert len(np.asarray(result["q_value"])) == n_loci


def test_sclvar_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # The docstring (and the function's own ValueError) require at least
    # 2 loci so that a genome-wide distribution exists to be an
    # outlier against. A single-locus input therefore must raise.
    counts = rng.integers(0, 50, (1, 4))
    n_total = [[50] * 4]
    with pytest.raises(ValueError):
        selection_coefficient(counts, n_total=n_total)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.sclvar as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
