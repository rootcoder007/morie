# morie.fn -- function file (rootcoder007/morie)
"""Random-forest out-of-bag error."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["random_forest_oob", "rf_oob_error"]


def random_forest_oob(y, predictions, in_bag=None, task="regression",
                      seed=0):
    r"""OOB error, and the learning curve that says whether to grow more.

    Each bootstrap sample omits about :math:`e^{-1} \approx 36.8\,\%`
    of the rows, so averaging the trees that excluded an observation
    gives a generalisation estimate at no cost in held-out data. In
    genomic prediction, where :math:`n` is small and :math:`p` is
    enormous, that is not a convenience -- it is often the only honest
    validation available.

    ``oob_curve`` recomputes the error using the first :math:`b` trees
    for increasing :math:`b`. A curve still falling at the final tree
    count means the forest is under-grown; a flat one means adding
    trees will not help, and the remaining error is bias or noise
    rather than variance. That distinction is the practical question
    and it cannot be answered from a single number.

    ``in_bag`` may be omitted, in which case bootstrap membership is
    simulated from ``seed`` -- useful for studying the estimator, but
    the real masks should be passed when they exist.

    Parameters
    ----------
    y : array-like, shape (n,)
    predictions : array-like, shape (B, n)
    in_bag : array-like of bool, shape (B, n), optional
    task : {'regression', 'classification'}
    seed : int

    Returns
    -------
    RichResult
        ``oob_error``, ``oob_r2``, ``oob_prediction``, ``oob_curve``,
        ``converged``, ``mean_oob_votes``, ``coverage``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), chapter 15,
    random forest for genomic prediction, pp. 633-640.
    Breiman (1996, 2001).

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.0, 1.0])
    >>> P = np.array([[9.0, 1.0], [0.0, 9.0]])
    >>> M = np.array([[True, False], [False, True]])
    >>> float(random_forest_oob(y, P, M)["oob_error"])
    0.0
    """
    yv = np.asarray(y, dtype=float).ravel()
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    n = yv.size
    if P.shape[1] != n:
        raise ValueError(
            "predictions must be (B, %d), got %s." % (n, P.shape)
        )
    B = P.shape[0]
    if task not in ("regression", "classification"):
        raise ValueError(
            "task must be 'regression' or 'classification', got %r." % task
        )
    if in_bag is None:
        rng = np.random.default_rng(int(seed))
        M = np.zeros((B, n), dtype=bool)
        for b in range(B):
            M[b, rng.integers(0, n, size=n)] = True
    else:
        M = np.atleast_2d(np.asarray(in_bag)).astype(bool)
        if M.shape != P.shape:
            raise ValueError(
                "in_bag must match predictions in shape, got %s and %s."
                % (M.shape, P.shape)
            )
    oob = ~M

    def _err(upto):
        o = oob[:upto]
        p = P[:upto]
        votes = o.sum(axis=0)
        have = votes > 0
        if not have.any():
            return np.nan, np.full(n, np.nan), have
        pred = np.full(n, np.nan)
        pred[have] = (np.where(o, p, 0.0).sum(axis=0)[have] / votes[have])
        if task == "classification":
            lab = (pred >= 0.5).astype(float)
            return float(np.mean(lab[have] != yv[have])), pred, have
        return float(np.mean((pred[have] - yv[have]) ** 2)), pred, have

    err, pred, have = _err(B)
    steps = sorted(set(
        [max(int(round(B * f)), 1) for f in (0.1, 0.25, 0.5, 0.75, 1.0)]
    ))
    curve = np.array([_err(b)[0] for b in steps])
    tail = curve[-2:] if curve.size > 1 else curve
    conv = bool(curve.size > 1
                and abs(tail[-1] - tail[0]) <= 0.02 * abs(tail[0] + 1e-12))
    var = float(np.var(yv[have])) if have.any() else np.nan
    return RichResult(
        payload={
            "estimate": err,
            "oob_error": err,
            "oob_r2": (float(1 - err / var)
                       if (task == "regression" and var > 0) else np.nan),
            "oob_accuracy": (float(1 - err)
                             if task == "classification" else np.nan),
            "oob_prediction": pred,
            "oob_curve": curve,
            "curve_at": np.asarray(steps),
            "converged": conv,
            "curve_note": (
                "a curve still falling at the final tree count means the "
                "forest is under-grown; a flat one means the remaining error "
                "is bias or noise, not variance, and more trees will not help"
            ),
            "coverage": float(np.mean(have)),
            "mean_oob_votes": float(oob.sum(axis=0).mean()),
            "expected_oob_fraction": float(np.exp(-1.0)),
            "bias_note": (
                "each row is scored only by the trees that excluded it, so "
                "the OOB error is mildly pessimistic for the full forest"
            ),
            "n_trees": int(B),
            "task": task,
            "n": int(n),
            "method": "Random-forest out-of-bag error",
        }
    )


def cheatsheet():
    return (
        "rfOOB: OOB error plus the learning curve that says whether growing "
        "more trees would help"
    )


#: Catalogue alias for :func:`random_forest_oob`.
rf_oob_error = random_forest_oob


# compact alias per ledger/NAMING.md
rfooberror = rf_oob_error
