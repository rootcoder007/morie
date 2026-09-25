"""Tests for multipsr.multi_stage_sampling (two-stage ratio estimator)."""

import math

import pytest

from morie.fn.multipsr import multi_stage_sampling

Y = [[3.0, 5.0, 4.0], [8.0, 7.0], [2.0, 4.0, 3.0, 5.0], [6.0, 6.5, 5.5]]
NL = [12.0, 9.0, 15.0, 10.0]
M, N = 10.0, 130.0


def test_multipsr_basic():
    """Yhat = sum N_l ybar_l / sum N_l and
    V = (M/N)^2 [ (M-m)/M / (m(m-1)) sum (N_l ybar_l - N_l Yhat)^2
                 + 1/(mM) sum N_l^2 (N_l - n_l)/N_l s_l^2 / n_l ],
    recomputed (samplingbook::submean, method = "ratio")."""
    result = multi_stage_sampling(Y, NL, M, N)
    assert isinstance(result, dict)
    m = len(Y)
    yb = [sum(y) / len(y) for y in Y]
    yhat = sum(n * b for n, b in zip(NL, yb)) / sum(NL)
    s2 = [sum((v - b) ** 2 for v in y) / (len(y) - 1) for y, b in zip(Y, yb)]
    between = (M - m) / M / (m * (m - 1)) * sum((n * b - n * yhat) ** 2 for n, b in zip(NL, yb))
    within = sum(n * n * (n - len(y)) / n * s / len(y) for n, y, s in zip(NL, Y, s2)) / (m * M)
    se = (M / N) * math.sqrt(between + within)
    assert result["estimate"] == pytest.approx(yhat, rel=1e-14)
    assert result["se"] == pytest.approx(se, rel=1e-13)


def test_multipsr_edge():
    """Taking every PSU (m = M) removes the between-PSU term; one PSU
    cannot give a variance."""
    r = multi_stage_sampling(Y, NL, 4.0, sum(NL))
    assert r["between_term"] == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        multi_stage_sampling(Y[:1], NL[:1], M, N)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest
import importlib as _importlib

_doctest_module = _importlib.import_module("morie.fn.multipsr")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
