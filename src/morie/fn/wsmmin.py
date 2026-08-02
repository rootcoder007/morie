# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimax risk over a finite decision problem."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_minimax"]


def wasserman_minimax(loss, estimator, family):
    """
    Minimax value of a finite estimator-vs-distribution game.

    Formula: R_minimax = inf_T sup_F R(T, F), computed exactly for a
    FINITE set of estimators (rows) and a finite family of
    distributions (columns) with the risk matrix supplied. Also
    reports the maximin value sup_F inf_T R(T, F) — by weak duality
    maximin <= minimax always, checked; equality certifies a
    saddle point in pure strategies.

    Parameters
    ----------
    loss : array-like, shape (m, k)
        Risk matrix R[i][j] = R(estimator_i, F_j).
    estimator : sequence
        Labels for the m estimators (rows).
    family : sequence
        Labels for the k distributions (columns).

    Returns
    -------
    result : dict
        Keys: estimate (minimax risk), minimax_estimator (label),
        worst_case (per estimator), maximin, has_pure_saddle,
        m, k, method.

    References
    ----------
    Wasserman (2004), Ch 12 (minimax theory).

    Examples
    --------
    >>> R = [[1.0, 4.0],
    ...      [2.0, 2.0],
    ...      [3.0, 1.0]]
    >>> out = wasserman_minimax(R, ["T1", "T2", "T3"], ["F1", "F2"])
    >>> out["estimate"]
    2.0
    >>> out["minimax_estimator"]
    'T2'
    >>> out["worst_case"]
    [4.0, 2.0, 3.0]
    >>> out["maximin"]     # column minima are (1, 1) -> maximin 1
    1.0
    >>> out["has_pure_saddle"]
    False
    >>> wasserman_minimax([[1.0, 2.0]], ["T1", "T2"], ["F1", "F2"])
    Traceback (most recent call last):
        ...
    ValueError: the risk matrix is 1x2 but there are 2 estimator labels.
    """
    R = np.atleast_2d(np.asarray(loss, dtype=float))
    m, k = R.shape
    est = list(estimator)
    fam = list(family)
    if len(est) != m:
        raise ValueError(f"the risk matrix is {m}x{k} but there are {len(est)} estimator labels.")
    if len(fam) != k:
        raise ValueError(f"the risk matrix is {m}x{k} but there are {len(fam)} family labels.")
    worst = np.max(R, axis=1)
    i_star = int(np.argmin(worst))
    minimax = float(worst[i_star])
    best = np.min(R, axis=0)
    maximin = float(np.max(best))
    if maximin > minimax + 1e-12:
        raise RuntimeError("weak duality violated — impossible; numerical fault.")
    return RichResult(payload={
        "estimate": minimax, "minimax_estimator": est[i_star],
        "worst_case": [float(v) for v in worst], "maximin": maximin,
        "has_pure_saddle": bool(abs(maximin - minimax) < 1e-12),
        "m": int(m), "k": int(k),
        "method": "exact min over rows of max over columns; maximin duality check"})


def cheatsheet():
    return "wsmmin: min_i max_j R[i,j]; maximin <= minimax checked; saddle flag"
