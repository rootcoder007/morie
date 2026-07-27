# morie.fn -- function file (rootcoder007/morie)
"""Pool-adjacent-violators isotonic regression for nonmetric MDS disparities."""

import numpy as np

from ._richresult import RichResult

__all__ = ["isotonic_regression_disparity"]


def _pava(y, w=None):
    """L2 isotonic (nondecreasing) fit by pool-adjacent-violators."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tot = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tot
            wts[-2] = tot
            cnts[-2] += cnts[-1]
            del vals[-1], wts[-1], cnts[-1]
    return np.repeat(vals, cnts)


def isotonic_regression_disparity(D, delta_rank):
    r"""Disparities for the nonmetric MDS loop.

    Orders the configuration distances by the rank of the observed
    dissimilarities and computes

    .. math:: \hat d = \arg\min_{\hat d \text{ monotone}}
              \sum (d_{ij} - \hat d_{ij})^2

    by pool-adjacent-violators -- the exact L2 projection onto the
    monotone cone (Kruskal's "monotone regression"). The result is the
    disparity vector in the original pair order.

    Parameters
    ----------
    D : array-like, shape (m,)
        Configuration distances per pair.
    delta_rank : array-like, shape (m,)
        Rank order of the observed dissimilarities (any orderable
        values; ties keep their block mean).

    Returns
    -------
    RichResult
        keys: ``disparities`` (m, original order), ``sorted_fit``,
        ``order``, ``n_pairs``, ``method``.

    References
    ----------
    Kruskal, J. B. (1964). Nonmetric multidimensional scaling: a
    numerical method. *Psychometrika*, 29(2), 115-129. (monotone
    regression inside the stress loop)
    """
    d = np.asarray(D, dtype=float).ravel()
    r = np.asarray(delta_rank, dtype=float).ravel()
    if d.size != r.size:
        raise ValueError("D and delta_rank must have equal length.")
    if d.size < 2:
        raise ValueError("need at least 2 pairs.")

    order = np.argsort(r, kind="stable")
    fit_sorted = _pava(d[order])
    disp = np.empty_like(d)
    disp[order] = fit_sorted

    return RichResult(
        payload={
            "disparities": disp,
            "sorted_fit": fit_sorted,
            "order": order,
            "n_pairs": int(d.size),
            "method": "Monotone (PAV) regression disparities for nonmetric MDS",
        }
    )


def cheatsheet():
    return "isotr: sort d by rank(delta), PAV project onto the monotone cone"
