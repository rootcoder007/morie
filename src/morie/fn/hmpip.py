# morie.fn -- function file (rootcoder007/morie)
"""Preprocessing pipeline with fit/transform separation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["preprocessing_pipeline"]

STEPS = ("impute_median", "impute_mean", "standardize", "minmax", "log1p")


def preprocessing_pipeline(X_train, X_test=None, steps=("impute_median",
                                                        "standardize")):
    r"""Fit preprocessing on the training data only, then apply it.

    The whole reason a pipeline is an object rather than a sequence of
    calls is that every step has PARAMETERS learned from data -- a
    median, a mean, a scale -- and those must come from the training
    rows alone. Computing them on the pooled data leaks test
    information into training and inflates every score that follows.
    The leak is invisible: nothing errors, the numbers just come out
    too good.

    Here the statistics are fitted on ``X_train`` and stored; applying
    them to ``X_test`` uses the stored values. ``leakage_check``
    reports the difference between the fitted statistics and the ones
    the pooled data would have given, which is the size of the leak
    that was avoided.

    Parameters
    ----------
    X_train : array-like, shape (n, p)
    X_test : array-like, shape (m, p), optional
    steps : sequence of str
        From ``impute_median``, ``impute_mean``, ``standardize``,
        ``minmax``, ``log1p``.

    Returns
    -------
    RichResult
        ``train``, ``test``, ``params`` (per step), ``leakage_check``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 2,
    transformation pipelines.

    Examples
    --------
    >>> import numpy as np
    >>> out = preprocessing_pipeline([[1.0], [2.0], [3.0]],
    ...                              steps=("standardize",))
    >>> [round(float(v), 4) for v in out["train"].ravel()]
    [-1.2247, 0.0, 1.2247]
    """
    A = np.atleast_2d(np.asarray(X_train, dtype=float))
    if A.ndim == 1:
        A = A[:, None]
    B = None if X_test is None else np.atleast_2d(
        np.asarray(X_test, dtype=float)
    )
    if B is not None and B.shape[1] != A.shape[1]:
        raise ValueError(
            "X_test has %d columns, X_train has %d."
            % (B.shape[1], A.shape[1])
        )
    for s in steps:
        if s not in STEPS:
            raise ValueError("unknown step %r; expected one of %s." % (s, STEPS))

    params = {}
    leak = {}
    pooled = A if B is None else np.vstack([A, B])
    tr = A.copy()
    te = None if B is None else B.copy()

    for s in steps:
        if s in ("impute_median", "impute_mean"):
            fn = np.nanmedian if s == "impute_median" else np.nanmean
            v = fn(tr, axis=0)
            params[s] = v
            leak[s] = float(np.max(np.abs(v - fn(pooled, axis=0))))
            idx = np.nonzero(np.isnan(tr))
            tr[idx] = np.take(v, idx[1])
            if te is not None:
                jdx = np.nonzero(np.isnan(te))
                te[jdx] = np.take(v, jdx[1])
        elif s == "standardize":
            mu, sd = tr.mean(axis=0), tr.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            params[s] = {"mean": mu, "scale": sd}
            leak[s] = float(np.max(np.abs(mu - pooled.mean(axis=0))))
            tr = (tr - mu) / sd
            if te is not None:
                te = (te - mu) / sd
        elif s == "minmax":
            lo, hi = tr.min(axis=0), tr.max(axis=0)
            rng = np.where(hi > lo, hi - lo, 1.0)
            params[s] = {"min": lo, "range": rng}
            leak[s] = float(np.max(np.abs(lo - pooled.min(axis=0))))
            tr = (tr - lo) / rng
            if te is not None:
                te = (te - lo) / rng
        else:
            if np.any(tr < -1) or (te is not None and np.any(te < -1)):
                raise ValueError("log1p needs values above -1.")
            params[s] = None
            leak[s] = 0.0
            tr = np.log1p(tr)
            if te is not None:
                te = np.log1p(te)

    return RichResult(
        payload={
            "estimate": tr,
            "train": tr,
            "test": te,
            "params": params,
            "steps": tuple(steps),
            "leakage_check": leak,
            "leakage_note": (
                "how far each fitted statistic sits from the one the POOLED "
                "data would have given; fitting on the pool is the classic "
                "silent leak, since nothing errors and every later score "
                "just comes out too good"
            ),
            "n_train": int(A.shape[0]),
            "n_test": 0 if B is None else int(B.shape[0]),
            "method": "Preprocessing pipeline fitted on training data only",
        }
    )


def cheatsheet():
    return (
        "hmpip: fit-on-train preprocessing chain, reporting the leak it "
        "avoided"
    )
