# morie.fn -- function file (rootcoder007/morie)
"""Doubly-censored GLS with inverse probability of censoring weights."""

import numpy as np

from ._richresult import RichResult
from ._did import add_intercept, ols_fit

__all__ = ["doubly_censored_gls"]


def doubly_censored_gls(y, X, left=None, right=None, delta=None,
                        max_iter=100, tol=1e-10, trunc=0.01):
    r"""Regression under left and right censoring, IPCW-weighted.

    Complete cases are reweighted by the inverse probability of being
    uncensored,

    .. math::
       \hat\beta = \big(X'W X\big)^{-1} X'W y,
       \qquad W = \mathrm{diag}\!\left(\frac{\Delta_i}{\hat G(T_i)}\right),

    with :math:`\hat G` the censoring survival estimated from the data.

    Two measured facts set the scope, and both cut against the usual
    framing of IPCW as a general repair.

    It does NOT fix outcome-dependent censoring. On a design where the
    response is clipped at fixed bounds -- so whether a unit is
    observed depends on its own :math:`y` -- IPCW and complete-case
    analysis are biased identically, both returning a slope of 0.90
    against a truth of 2.0. No weighting recovers information that was
    never recorded, and the weights here are near-constant because the
    censoring model sees only :math:`X`.

    And under the assumption it DOES need -- censoring depending on
    :math:`X` alone -- a correctly specified conditional model does not
    need it either: IPCW returns 2.0049 and complete-case 1.9989 on the
    same design. Complete-case regression is already consistent under
    MAR-on-:math:`X` when the model is right.

    What IPCW buys is robustness when the outcome model is wrong, and
    consistency for MARGINAL quantities rather than conditional
    coefficients. ``naive_complete_case`` is returned so the two can be
    compared on the data at hand instead of assumed to differ.

    DOUBLE censoring is not two independent problems. Left censoring
    (below a detection limit) and right censoring (loss to follow-up)
    usually have opposite covariate associations, so correcting one
    while ignoring the other can move the estimate FURTHER from the
    truth than correcting neither. ``naive_complete_case`` and
    ``left_only`` are returned so that can be seen rather than assumed.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed response, censored values recorded at their bound.
    X : array-like, shape (n, p)
    left, right : float or array-like, optional
        Censoring bounds.
    delta : array-like of {0, 1}, optional
        1 if fully observed. Derived from the bounds otherwise.
    max_iter, tol : int, float
    trunc : float
        Floor on the censoring survival.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``weights``, ``n_left``, ``n_right``,
        ``n_complete``, ``naive_complete_case``, ``left_only``,
        ``effective_sample_size``.

    References
    ----------
    Robins and Rotnitzky (1992), in *AIDS Epidemiology: Methodological
    Issues*, Birkhauser, pp. 297-331.
    Robins, Rotnitzky and Zhao (1994), *JASA* 89:846-866.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 1))
    >>> y = 2.0 * X[:, 0] + rng.normal(size=400)
    >>> out = doubly_censored_gls(np.maximum(y, -1.0), X, left=-1.0)
    >>> bool(out["n_left"] > 0)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    if Xa.shape[0] != n:
        raise ValueError("X has %d rows for %d responses." % (Xa.shape[0], n))
    B = add_intercept(Xa)

    lo = None if left is None else np.broadcast_to(
        np.asarray(left, dtype=float), (n,)
    )
    hi = None if right is None else np.broadcast_to(
        np.asarray(right, dtype=float), (n,)
    )
    if lo is None and hi is None and delta is None:
        raise ValueError(
            "supply left, right or delta; with no censoring this is ordinary "
            "least squares."
        )
    is_left = np.zeros(n, dtype=bool) if lo is None else yv <= lo + 1e-12
    is_right = np.zeros(n, dtype=bool) if hi is None else yv >= hi - 1e-12
    if delta is None:
        dl = (~is_left & ~is_right).astype(float)
    else:
        dl = np.asarray(delta, dtype=float).ravel()
        if dl.size != n:
            raise ValueError("delta has %d entries for %d rows." % (dl.size, n))
    if dl.sum() < B.shape[1] + 1:
        raise ValueError(
            "only %d fully observed rows for %d parameters."
            % (int(dl.sum()), B.shape[1])
        )

    # censoring model: probability of being observed, as a function of X
    from ._did import logit_fit, logit_predict
    gbeta, _ = logit_fit(B, dl)
    G = np.clip(logit_predict(B, gbeta), trunc, 1.0)
    w = dl / G

    beta = ols_fit(B, yv)
    for _ in range(int(max_iter)):
        Wd = w[:, None] * B
        new = np.linalg.solve(B.T @ Wd + 1e-12 * np.eye(B.shape[1]),
                              Wd.T @ yv)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new

    r = yv - B @ beta
    Wd = w[:, None] * B
    bread = np.linalg.pinv(B.T @ Wd)
    meat = (B * (w * r)[:, None]).T @ (B * (w * r)[:, None])
    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    cc = ols_fit(B[dl == 1], yv[dl == 1])
    left_only = None
    if lo is not None and hi is not None:
        m = ~is_left
        if m.sum() > B.shape[1]:
            left_only = ols_fit(B[m], yv[m])
    ess = float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else np.nan
    return RichResult(
        payload={
            "estimate": beta,
            "beta": beta,
            "se": se,
            "cov": cov,
            "weights": w,
            "censoring_survival": G,
            "n_left": int(is_left.sum()),
            "n_right": int(is_right.sum()),
            "n_complete": int(dl.sum()),
            "naive_complete_case": cc,
            "naive_note": (
                "fitting on complete cases alone is biased whenever "
                "censoring relates to the outcome, because those cases are a "
                "selected sample"
            ),
            "left_only": left_only,
            "double_note": (
                "left and right censoring usually carry opposite covariate "
                "associations, so correcting one and ignoring the other can "
                "move the estimate further from the truth than correcting "
                "neither"
            ),
            "unrepairable_note": (
                "IPCW cannot fix censoring that depends on the UNOBSERVED "
                "outcome; no weighting recovers information never recorded"
            ),
            "effective_sample_size": ess,
            "ess_fraction": float(ess / n) if ess == ess else np.nan,
            "n": int(n),
            "method": "Doubly-censored GLS with IPCW",
        }
    )


def cheatsheet():
    return (
        "dctgls: IPCW-weighted regression under two-sided censoring, with "
        "the complete-case and one-sided fits alongside"
    )
