# morie.fn -- function file (rootcoder007/morie)
"""Sure independence screening -- Fan & Lv (2008), ESL Sec 18.3."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_sis_screening"]


def esl_sis_screening(X, y, d=None):
    r"""Rank predictors by marginal correlation and keep the top ``d``.

    .. math::
        \omega_j = \left| \operatorname{corr}(X_j, y) \right|,

    keeping the ``d`` largest. Fan & Lv's sure screening property says that
    under conditions on the design, the retained set contains the true
    active set with probability tending to one -- which is what licenses
    using this as a *pre-filter* before a proper selection method, not as a
    selection method itself.

    The standard default is :math:`d = n - 1`, or :math:`d = \lfloor n/\log n
    \rfloor` in the original paper. The known failure mode is a predictor
    that matters only jointly: an important variable marginally uncorrelated
    with ``y`` is screened out and never recovered, which is exactly what
    iterated SIS was introduced to patch.

    Constant columns have undefined correlation and are ranked last rather
    than producing a NaN.

    Parameters
    ----------
    X : array-like
        Predictors ``(n, p)``.
    y : array-like
        Response ``(n,)``.
    d : int, optional
        Number to keep. Defaults to ``min(p, n - 1)``.

    Returns
    -------
    RichResult
        ``selected`` (column indices, strongest first), ``omega`` (the
        marginal statistics), ``rank``, ``d``, ``dropped``.

    References
    ----------
    Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh
        dimensional feature space. *JRSS-B*, 70(5), 849-911.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    The two genuinely marginal predictors are retained out of 50.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 50))
    >>> y = 3 * X[:, 7] - 2 * X[:, 21] + rng.normal(0, 0.1, 100)
    >>> sel = esl_sis_screening(X, y, d=5)["selected"]
    >>> bool(7 in sel and 21 in sel)
    True

    A constant column ranks last instead of yielding NaN.

    >>> Xc = np.column_stack([X, np.ones(100)])
    >>> r = esl_sis_screening(Xc, y, d=51)
    >>> int(r["rank"][50])
    50

    The documented blind spot. For ``y = z * w`` both inputs are essential,
    yet each is marginally uncorrelated with ``y``, so their screening
    statistics sit down among the pure-noise columns -- here ``w`` ranks
    below 15 of the 20 noise predictors.

    >>> z = rng.normal(size=200)
    >>> w = rng.normal(size=200)
    >>> Xi = np.column_stack([z, w, rng.normal(size=(200, 20))])
    >>> yi = z * w
    >>> r = esl_sis_screening(Xi, yi, d=3)
    >>> bool(r["omega"][1] < np.median(r["omega"][2:]))
    True

    Contrast the marginal case above, where the true predictors' statistics
    stand far clear of the noise.

    >>> rs = esl_sis_screening(X, y, d=5)
    >>> bool(rs["omega"][7] > 0.5 and r["omega"][:2].max() < 0.25)
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if n != y.size:
        raise ValueError(f"X has {n} rows but y has {y.size}")
    if n < 3:
        raise ValueError("need at least 3 observations to screen")
    d = min(p, n - 1) if d is None else int(d)
    if not 1 <= d <= p:
        raise ValueError(f"d must be between 1 and p={p}")

    yc = y - y.mean()
    sy = float(np.sqrt(np.sum(yc**2)))
    Xc = X - X.mean(axis=0)
    sx = np.sqrt((Xc**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.abs((Xc * yc[:, None]).sum(axis=0) / (sx * sy))
    omega = np.where(np.isfinite(omega), omega, -np.inf)

    order = np.argsort(-omega, kind="stable")
    rank = np.empty(p, dtype=int)
    rank[order] = np.arange(p)
    selected = order[:d]
    return RichResult(
        title="Sure independence screening",
        summary_lines=[("n", n), ("p", p), ("kept", int(d))],
        payload={
            "selected": selected,
            "omega": np.where(np.isfinite(omega), omega, np.nan),
            "rank": rank, "d": int(d),
            "dropped": order[d:], "n": int(n), "p": int(p),
            "method": "esl_sis_screening",
        },
    )


def cheatsheet():
    return "eslsis: marginal |corr| screen; a pre-filter, not a selector -- interaction-only predictors are missed"
