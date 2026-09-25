"""Tests for prsccs.prs_cs (PRS-CS posterior mean, Ge et al. 2019)."""

import pytest

from morie.fn.prsccs import prs_cs

BH = [0.12, -0.05, 0.30]
D = [[1.0, 0.4, 0.1], [0.4, 1.0, 0.3], [0.1, 0.3, 1.0]]
PSI = [0.5, 0.02, 2.0]


def _solve3(A, b):
    import copy
    M = [row[:] + [v] for row, v in zip(copy.deepcopy(A), b)]
    for c in range(3):
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][3] / M[i][i] for i in range(3)]


def test_prsccs_basic():
    """beta_post = (D + Psi^-1)^-1 beta_hat, recomputed; n and sigma2 do
    not enter the mean (the PRS-CS sampler's ld_blk + diag(1/psi))."""
    A = [[D[i][j] + (1.0 / PSI[i] if i == j else 0.0) for j in range(3)] for i in range(3)]
    exp = _solve3(A, BH)
    r = prs_cs(BH, D, PSI, n=5000)
    assert isinstance(r, dict)
    assert r["beta"] == pytest.approx(exp, rel=1e-13)
    assert prs_cs(BH, D, PSI, n=50)["beta"] == pytest.approx(exp, rel=1e-13)


def test_prsccs_edge():
    """A tiny local variance shrinks its SNP hard towards zero; a
    non-positive psi is refused."""
    r = prs_cs(BH, D, PSI, n=5000)
    assert abs(r["shrinkage"][1]) < abs(r["shrinkage"][2])
    with pytest.raises(ValueError):
        prs_cs(BH, D, [0.5, 0.0, 2.0], n=100)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest
import importlib as _importlib

_doctest_module = _importlib.import_module("morie.fn.prsccs")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
