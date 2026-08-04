# morie.fn -- function file (rootcoder007/morie)
"""Cronbach's alpha with each item deleted in turn.

Source: Cronbach, L. J. (1951), "Coefficient alpha and the internal
structure of tests", *Psychometrika* 16(3):297-334, coefficient alpha as

    alpha = (k / (k - 1)) * (1 - sum_i V_i / V_t)

applied k times, once to each of the k sub-scales obtained by dropping a
single item.  "alpha if item deleted" is not a separate estimator: it is
alpha itself, recomputed on k - 1 items, and every quantity below is
that recomputation.  The largest of the k values, and the item whose
removal produces it, are reported as ``max_alpha`` / ``argmax_alpha``.

An item whose deletion RAISES alpha is the diagnostic this table exists
for, so ``alpha_full`` (alpha on all k items) is returned alongside, and
``delta`` gives alpha_dropped - alpha_full item by item.

The variance convention, and the reason no standard error is offered,
are as in ``ctta1c``; the two modules share one definition of alpha.
"""

from ._richresult import RichResult

__all__ = ["ctt_alpha_max"]


def _alpha_on(rows, n, cols):
    """Cronbach alpha over the given column indices of an n-row table."""
    k = len(cols)
    if k < 2:
        raise ValueError("alpha needs at least two items")
    sv = 0.0
    for j in cols:
        s = 0.0
        for i in range(n):
            s += rows[i][j]
        m = s / n
        ss = 0.0
        for i in range(n):
            dv = rows[i][j] - m
            ss += dv * dv
        sv += ss / (n - 1)
    tot = []
    for i in range(n):
        t = 0.0
        for j in cols:
            t += rows[i][j]
        tot.append(t)
    m = sum(tot) / n
    ss = 0.0
    for v in tot:
        dv = v - m
        ss += dv * dv
    tv = ss / (n - 1)
    if tv <= 0.0:
        raise ValueError("total score has zero variance; alpha is undefined")
    return (k / (k - 1.0)) * (1.0 - sv / tv)


def ctt_alpha_max(X):
    """Alpha-if-item-deleted for an n persons x k items score matrix.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Rows are persons, columns are items.  At least three items are
        needed, since dropping one must leave a two-item scale.

    Returns
    -------
    RichResult
        ``alpha_full``, ``alpha_dropped`` (length k), ``delta``,
        ``max_alpha``, ``argmax_alpha``, ``estimate`` (= ``max_alpha``),
        ``n_items``, ``n``.
    """
    rows = [[float(v) for v in row] for row in X]
    n = len(rows)
    if n < 2:
        raise ValueError("alpha needs at least two persons")
    k = len(rows[0])
    if any(len(r) != k for r in rows):
        raise ValueError("every person must be scored on the same items")
    if k < 3:
        raise ValueError("alpha-if-item-deleted needs at least three items")
    full = _alpha_on(rows, n, list(range(k)))
    drop = []
    for j in range(k):
        cols = [c for c in range(k) if c != j]
        drop.append(_alpha_on(rows, n, cols))
    best = 0
    for j in range(1, k):
        if drop[j] > drop[best]:
            best = j
    return RichResult(payload={
        "alpha_full": float(full),
        "alpha_dropped": [float(v) for v in drop],
        "delta": [float(v - full) for v in drop],
        "max_alpha": float(drop[best]), "argmax_alpha": best,
        "estimate": float(drop[best]), "n_items": k, "n": n,
        "method": "Cronbach (1951) alpha recomputed with each item deleted"})


def cheatsheet():
    return "cttamx: Cronbach alpha with each item deleted in turn"


# compact alias per ledger/NAMING.md
cttalphamax = ctt_alpha_max
