# morie.fn -- function file (rootcoder007/morie)
"""Angle-based outlier detection -- Kriegel et al. (2008)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["abod"]


def abod(X, k=None):
    r"""Score outliers by the variance of angles to pairs of other points.

    For each point :math:`p`, take the weighted variance over pairs
    :math:`(a, b)` of the cosine

    .. math::
        \frac{\langle a - p,\; b - p\rangle}
             {\lVert a-p\rVert^2 \, \lVert b-p\rVert^2},

    An interior point sees other points in every direction, so the angle
    spectrum is broad and the variance is large. An outlier sees the whole
    cloud in roughly one direction, so the angles cluster and the variance
    collapses -- **low ABOF means outlying**, the opposite polarity to most
    scores, and ``score`` is returned as the negated ABOF so that higher is
    outlying either way.

    Angles rather than distances is what makes this hold up in high dimension,
    where distance concentration makes every pair look equidistant and
    distance-based scores lose contrast.

    Exact ABOD is :math:`O(n^3)`. With ``k`` set, the faster approximation
    over each point's ``k`` nearest neighbours is used.

    Parameters
    ----------
    X : array-like
        Data ``(n, d)``.
    k : int, optional
        Neighbourhood size for the approximation. ``None`` computes the exact
        cubic version, which is fine to a few hundred points.

    Returns
    -------
    RichResult
        ``abof`` (low is outlying), ``score`` (``-abof``, high is outlying),
        ``rank``.

    References
    ----------
    Kriegel, H.-P., Schubert, M., & Zimek, A. (2008). Angle-based outlier
        detection in high-dimensional data. *KDD 2008*, 444-452.

    Examples
    --------
    The outlier has the smallest angle variance, hence the largest score.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(0, 1, (120, 3)), [[10.0, 10.0, 10.0]]]
    >>> r = abod(X)
    >>> int(np.argmax(r["score"]))
    120
    >>> bool(r["abof"][120] < np.median(r["abof"][:120]))
    True

    The approximation agrees on the ranking.

    >>> int(np.argmax(abod(X, k=30)["score"]))
    120

    >>> abod(np.zeros((2, 2)))
    Traceback (most recent call last):
        ...
    ValueError: need at least 3 points to form an angle
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points to form an angle")
    if k is not None:
        k = int(k)
        if not 2 <= k <= n - 1:
            raise ValueError(f"k must be between 2 and {n - 1}")
        D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0.0))
        np.fill_diagonal(D, np.inf)
        nbrs = np.argsort(D, axis=1, kind="stable")[:, :k]

    abof = np.empty(n)
    for i in range(n):
        idx = nbrs[i] if k is not None else np.setdiff1d(np.arange(n), i)
        V = X[idx] - X[i]
        nrm2 = (V**2).sum(axis=1)
        good = nrm2 > 1e-24
        V, nrm2 = V[good], nrm2[good]
        if V.shape[0] < 2:
            abof[i] = 0.0
            continue
        G = V @ V.T
        w = 1.0 / np.outer(nrm2, nrm2)
        vals = G * w
        iu = np.triu_indices(V.shape[0], 1)
        vv, ww = vals[iu], w[iu]
        wsum = ww.sum()
        mean = float((ww * vv).sum() / wsum)
        abof[i] = float((ww * (vv - mean) ** 2).sum() / wsum)

    score = -abof
    order = np.argsort(-score, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return RichResult(
        title="Angle-based outlier detection",
        summary_lines=[("n", n), ("d", int(X.shape[1])),
                       ("mode", "approximate" if k else "exact")],
        payload={
            "abof": abof, "score": score, "rank": rank, "k": k,
            "method": "abod",
        },
    )


def cheatsheet():
    return "abod: LOW angle variance = outlying (score negates it); angles beat distances in high d"
