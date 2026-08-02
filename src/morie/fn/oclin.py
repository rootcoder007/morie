# morie.fn -- function file (rootcoder007/morie)
"""Optimal Classification cutting line for a single roll call."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["oc_cutting_line"]


def oc_cutting_line(ideal_points, votes):
    r"""Error-minimising cutting line given fixed ideal points.

    In one dimension this is Poole's exhaustive-search step: for a
    rank-ordered set of legislators, try every cutpoint between
    adjacent ideal points (both polarities) and keep the one that
    minimises classification errors -- the "given a rank order of
    legislators, the global maximum in correct classification can be
    found for every roll call" step of the OC algorithm. In higher
    dimensions the normal vector is estimated by linear-discriminant
    projection and the same 1-D search runs along it.

    Parameters
    ----------
    ideal_points : array-like, shape (n,) or (n, k)
        Legislator coordinates.
    votes : array-like, shape (n,)
        Binary votes on the roll call (1 = yea, 0 = nay, NaN skipped).

    Returns
    -------
    RichResult
        keys: ``cutpoint`` (scalar position along ``normal``),
        ``normal`` (k, unit vector), ``polarity`` (1 = yeas above the
        cut), ``errors``, ``correct_classification``, ``predicted``
        (n, with NaN where the vote was missing), ``n``, ``method``.

    References
    ----------
    Poole, K. T. (2000). Nonparametric unfolding of binary choice
    data. *Political Analysis*, 8(3), 211-237. (the cutting-plane /
    Janice algorithm; library PDF, pp. 211-213 verified)
    """
    X = np.atleast_2d(np.asarray(ideal_points, dtype=float))
    if X.shape[0] == 1 and np.ndim(ideal_points) == 1:
        X = X.T
    v = np.asarray(votes, dtype=float).ravel()
    n, k = X.shape
    if v.size != n:
        raise ValueError("votes must have one entry per legislator.")
    valid = ~np.isnan(v)
    if valid.sum() < 2 or len(np.unique(v[valid])) < 2:
        raise ValueError("need at least one yea and one nay among non-missing votes.")

    if k == 1:
        normal = np.array([1.0])
        proj = X[:, 0]
    else:
        # discriminant direction: difference of class means, whitened
        y = v[valid]
        Xv = X[valid]
        mu1 = Xv[y == 1].mean(axis=0)
        mu0 = Xv[y == 0].mean(axis=0)
        S = np.cov(Xv.T) + 1e-8 * np.eye(k)
        w = np.linalg.solve(S, mu1 - mu0)
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            raise ValueError("degenerate geometry: class means coincide.")
        normal = w / norm
        proj = X @ normal

    order = np.argsort(proj[valid])
    pv = proj[valid][order]
    yv = v[valid][order]
    cuts = np.concatenate([[pv[0] - 1.0], (pv[1:] + pv[:-1]) / 2.0, [pv[-1] + 1.0]])

    best = (np.inf, cuts[0], 1)
    for c in cuts:
        above = pv > c
        e_pos = int((yv[above] == 0).sum() + (yv[~above] == 1).sum())
        e_neg = int((yv[above] == 1).sum() + (yv[~above] == 0).sum())
        if e_pos < best[0]:
            best = (e_pos, c, 1)
        if e_neg < best[0]:
            best = (e_neg, c, -1)

    errors, cut, pol = best
    pred = np.full(n, np.nan)
    side = proj > cut
    pred[valid] = np.where(side[valid], 1.0 if pol == 1 else 0.0, 0.0 if pol == 1 else 1.0)

    return RichResult(
        payload={
            "cutpoint": float(cut),
            "normal": normal,
            "polarity": int(pol),
            "errors": int(errors),
            "correct_classification": float(1 - errors / valid.sum()),
            "predicted": pred,
            "n": int(n),
            "method": "OC cutting line (exhaustive cutpoint search along the discriminant)",
        }
    )


def cheatsheet():
    return "oclin: try every cutpoint between adjacent ideal points, both polarities (Poole 2000)"
