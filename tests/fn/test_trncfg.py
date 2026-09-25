"""Tests for trncfg.truncated_cf_estimator (Crump et al. 2009 trimming)."""

import math
import statistics

import pytest

from morie.fn.trncfg import truncated_cf_estimator


def _data(n=300, s=5.0):
    x = [s * math.sin(1.37 * k) for k in range(n)]
    e = [1 / (1 + math.exp(-v)) for v in x]
    d = [1.0 if ((k * 37 + 11) % 97 + 0.5) / 97 < p else 0.0 for k, p in enumerate(e)]
    y = [a + v + 0.3 * math.cos(2.3 * k) for k, (a, v) in enumerate(zip(d, x))]
    return y, d, e


def _crump_alpha(e):
    """Corollary 1, sample version: the smallest alpha with
    1/(alpha(1-alpha)) <= 2 mean(inv | inv <= 1/(alpha(1-alpha))),
    found by scanning alpha on a fine grid (step 1e-6) -- the kept set
    it implies is then compared, which is insensitive to the grid."""
    inv = [1 / (p * (1 - p)) for p in e]
    if max(inv) <= 2 * statistics.fmean(inv):
        return 0.0
    a = 1e-6
    while a < 0.5:
        g = 1 / (a * (1 - a))
        kept = [v for v in inv if v <= g]
        if kept and g <= 2 * statistics.fmean(kept):
            return a
        a += 1e-6
    return None


def test_trncfg_basic():
    """The kept set is Crump's optimal one (PSweight::PStrim(optimal =
    TRUE) keeps the same 98 units on this design); the estimate is the
    Hajek IPW contrast on it."""
    y, d, e = _data()
    a = _crump_alpha(e)
    keep = [a <= p <= 1 - a for p in e]
    r = truncated_cf_estimator(y, d, propensity=e)
    kept_mod = [r["alpha"] <= p <= 1 - r["alpha"] for p in e]
    assert kept_mod == keep
    assert r["n_kept"] == sum(keep) == 98
    idx = [i for i in range(len(y)) if keep[i]]
    w1 = [d[i] / e[i] for i in idx]
    w0 = [(1 - d[i]) / (1 - e[i]) for i in idx]
    est = (sum(w * y[i] for w, i in zip(w1, idx)) / sum(w1)
           - sum(w * y[i] for w, i in zip(w0, idx)) / sum(w0))
    assert r["estimate"] == pytest.approx(est, abs=1e-12)
    assert r["n_dropped"] == len(y) - 98


def test_trncfg_edge():
    """Good overlap trims nothing (sup inv <= 2 E inv); the fixed rule
    uses 0.1; non-binary d and a missing propensity model raise."""
    y, d, e = _data(s=1.0)
    assert truncated_cf_estimator(y, d, propensity=e)["alpha"] == 0.0
    assert truncated_cf_estimator(y, d, propensity=e, rule="fixed")["alpha"] == 0.1
    with pytest.raises(ValueError):
        truncated_cf_estimator(y, [0.5] * len(y), propensity=e)
    with pytest.raises(ValueError):
        truncated_cf_estimator(y, d)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.trncfg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
