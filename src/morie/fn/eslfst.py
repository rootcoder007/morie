# morie.fn -- slice k04 (rootcoder007/morie)
"""F-test for a nested pair of linear models -- ESL eq (3.13).

Source READ FROM THE CORPUS PDF: Hastie, Tibshirani and Friedman,
*The Elements of Statistical Learning* (2nd ed., 2009), section 3.2,
equation (3.13), quoted verbatim from the corpus copy
``BookAdvanced_elementsofstatisticallearning.pdf``::

    F = (RSS0 - RSS1) / (p1 - p0)
        -------------------------
             RSS1 / (N - p1 - 1)

where RSS1 is the residual sum-of-squares of the bigger model with
p1 + 1 parameters and RSS0 that of the nested smaller model with
p0 + 1 parameters.  Under the Gaussian null F ~ F(p1 - p0, N - p1 - 1).

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["esl_f_test"]


def _rss(X, y, cols):
    n = y.shape[0]
    cols = list(cols)
    if cols:
        D = np.column_stack([np.ones(n)] + [X[:, int(j)] for j in cols])
    else:
        D = np.ones((n, 1))
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ beta
    return float(r @ r), len(cols)


def esl_f_test(model0, model1, X, y):
    """F-test of the nested model ``model0`` inside ``model1``.

    Parameters
    ----------
    model0, model1 : sequence of int
        Column indices of ``X`` in each model; ``model0`` must be a
        subset of ``model1``.  An intercept is always added.
    X : array-like, shape (N, p)
        Predictor matrix without an intercept column.
    y : array-like, shape (N,)
        Response.

    Returns
    -------
    RichResult
        keys: ``statistic``, ``p_value``, ``df1``, ``df2``, ``rss0``,
        ``rss1``, ``p0``, ``p1``, ``n``, ``method``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)
    s0 = set(int(j) for j in model0)
    s1 = set(int(j) for j in model1)
    if not s0 <= s1:
        raise ValueError("model0 must be nested inside model1")
    rss0, p0 = _rss(X, y, sorted(s0))
    rss1, p1 = _rss(X, y, sorted(s1))
    df1 = p1 - p0
    df2 = n - p1 - 1
    if df1 <= 0 or df2 <= 0:
        raise ValueError("need p1 > p0 and N > p1 + 1")
    stat = ((rss0 - rss1) / df1) / (rss1 / df2)
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(stats.f.sf(stat, df1, df2)),
            "df1": int(df1),
            "df2": int(df2),
            "rss0": rss0,
            "rss1": rss1,
            "p0": int(p0),
            "p1": int(p1),
            "n": n,
            "method": "F-test for nested linear models (ESL eq. 3.13)",
        }
    )


def cheatsheet():
    return "eslfst: nested-model F test (ESL eq. 3.13)"


# compact alias per ledger/NAMING.md
eslftest = esl_f_test
