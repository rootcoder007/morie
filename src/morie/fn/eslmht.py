# morie.fn -- slice k04 (rootcoder007/morie)
"""Holm (1979) step-down multiple-testing procedure.

Source: Holm, S. (1979).  A simple sequentially rejective multiple test
procedure.  *Scandinavian Journal of Statistics* 6, 65-70.  The 1979
paper is paywalled here; the procedure is quoted in its standard
published form, which is unambiguous and is the form given in Hastie,
Tibshirani and Friedman, *The Elements of Statistical Learning*
(2nd ed., 2009), section 18.7:

    order p_(1) <= p_(2) <= ... <= p_(m);
    let L be the smallest j with p_(j) > alpha / (m - j + 1);
    reject H_(1), ..., H_(L-1) and no others.

The monotone step-down adjusted p-values follow from the same rule:

    ptilde_(j) = max_{k <= j} min(1, (m - k + 1) p_(k)).

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_holm_bonferroni"]


def esl_holm_bonferroni(pvalues, alpha=0.05):
    """Holm step-down rejections and adjusted p-values.

    Parameters
    ----------
    pvalues : array-like
        The m raw p-values, in the caller own order.
    alpha : float, default 0.05
        Family-wise error rate.

    Returns
    -------
    RichResult
        keys: ``reject`` (bool array in input order), ``p_adjusted``
        (in input order), ``n_reject``, ``alpha``, ``m``, ``method``.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = int(p.size)
    if m == 0:
        raise ValueError("pvalues is empty")
    alpha = float(alpha)
    order = np.argsort(p)
    ps = p[order]

    k = m
    for j in range(m):
        if ps[j] > alpha / (m - j):
            k = j
            break
    reject_sorted = np.array([j < k for j in range(m)], dtype=bool)

    adj = np.empty(m, dtype=float)
    run = 0.0
    for j in range(m):
        run = max(run, min(1.0, (m - j) * float(ps[j])))
        adj[j] = run

    reject = np.empty(m, dtype=bool)
    p_adj = np.empty(m, dtype=float)
    reject[order] = reject_sorted
    p_adj[order] = adj
    return RichResult(
        payload={
            "reject": reject,
            "p_adjusted": p_adj,
            "n_reject": int(k),
            "alpha": alpha,
            "m": m,
            "method": "Holm (1979) step-down multiple test",
        }
    )


def cheatsheet():
    return "eslmht: Holm step-down multiple testing (Holm 1979)"
