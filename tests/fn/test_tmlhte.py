"""Tests for tmlhte.tmle_heterogeneous."""

import math

import pytest

from morie.fn._tmle import tmle_ate
from morie.fn.tmlhte import tmle_heterogeneous


def _data(n=600):
    W = [math.sin(1.7 * k) for k in range(n)]
    s = [k % 3 for k in range(n)]
    A = [1.0 if ((37 * k + 11) % 97 + 0.5) / 97.0 < 0.4 + 0.2 * (w > 0) else 0.0 for k, w in enumerate(W)]
    y = [0.3 + 0.1 * w + a * (0.1 + 0.15 * t) + 0.1 * math.sin(13.3 * k)
         for k, (w, a, t) in enumerate(zip(W, A, s))]
    return y, A, W, s


def test_tmlhte_basic():
    """Each stratum is its own tmle_ate fit; the pooled estimate is the
    inverse-variance mean; Cochran's Q with K-1 = 2 degrees of freedom has
    the closed-form upper tail exp(-Q/2)."""
    y, A, W, s = _data()
    fits = []
    for lab in (0, 1, 2):
        idx = [i for i in range(len(y)) if s[i] == lab]
        f = tmle_ate([y[i] for i in idx], [A[i] for i in idx], [[W[i]] for i in idx], trunc=0.01)
        fits.append((float(f["ate"]), float(f["se"])))
    w = [1.0 / se ** 2 for _, se in fits]
    pooled = sum(wi * e for wi, (e, _) in zip(w, fits)) / sum(w)
    q = sum(wi * (e - pooled) ** 2 for wi, (e, _) in zip(w, fits))
    r = tmle_heterogeneous(y, A, W, s)
    for lab, (e, se) in zip((0, 1, 2), fits):
        assert r["by_stratum"][lab]["estimate"] == pytest.approx(e, abs=1e-12)
        assert r["by_stratum"][lab]["se"] == pytest.approx(se, rel=1e-12)
    assert r["estimate"] == pytest.approx(pooled, abs=1e-12)
    assert r["se"] == pytest.approx(1.0 / math.sqrt(sum(w)), rel=1e-12)
    assert r["heterogeneity_q"] == pytest.approx(q, rel=1e-12)
    assert r["heterogeneity_df"] == 2
    assert r["heterogeneity_p"] == pytest.approx(math.exp(-q / 2), rel=1e-9)
    assert r["i_squared"] == pytest.approx(max(0.0, (q - 2) / q), rel=1e-12)


def test_tmlhte_edge():
    """Non-binary treatment and a single stratum raise; a small stratum
    and a one-arm stratum are dropped with a reason."""
    y, A, W, s = _data()
    with pytest.raises(ValueError):
        tmle_heterogeneous(y, [0.5] * len(y), W, s)
    with pytest.raises(ValueError):
        tmle_heterogeneous(y, A, W, [0] * len(y))
    s2 = [9 if k < 10 else t for k, t in enumerate(s)]
    A2 = [1.0 if t == 2 else a for a, t in zip(A, s)]
    r = tmle_heterogeneous(y, A2, W, s2)
    assert set(r["dropped"]) == {9, 2}
    assert r["n_strata"] == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.tmlhte as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
