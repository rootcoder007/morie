# morie.fn -- function file (rootcoder007/morie)
"""Doubly-robust survey estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["survey_dr_estimator"]


def survey_dr_estimator(y, D, X, sampling_weights=None, ps=None, mu1=None,
                        mu0=None):
    r"""Doubly-robust estimator under a sampling design:

    .. math:: \hat\tau = \frac{1}{\sum w}\sum_i w_i\left[
              \hat\mu_1(X_i) - \hat\mu_0(X_i)
              + \frac{D_i\{Y_i - \hat\mu_1\}}{\hat e(X_i)}
              - \frac{(1-D_i)\{Y_i - \hat\mu_0\}}{1 - \hat e(X_i)}
              \right].

    Consistent if EITHER the propensity model or the outcome model is
    right -- not both, and not neither. That is the whole claim, and
    it is worth being precise about what it does not say: two wrong
    models do not rescue each other, and double robustness gives no
    protection against unmeasured confounding at all.

    Under a complex design the sampling weights multiply the
    influence function, which is why they appear outside rather than
    inside the augmentation -- the design randomises WHO is observed,
    the treatment mechanism randomises what they receive, and the two
    are separate sources of randomness.

    Estimated propensities near 0 or 1 make the augmentation terms
    explode, exactly as in :mod:`morie.fn.survipw`, so the extreme
    propensity is reported.

    Parameters
    ----------
    y : array-like
        Outcomes.
    D : array-like of {0, 1}
        Treatment indicators.
    X : array-like
        Covariates, used to fit working models when none are given.
    sampling_weights : array-like, optional
        Design weights.
    ps : array-like, optional
        Propensity scores; a logistic fit otherwise.
    mu1, mu0 : array-like, optional
        Outcome predictions; linear fits otherwise.

    Returns
    -------
    RichResult
        keys: ``ate``, ``influence``, ``se``, ``min_ps``, ``max_ps``,
        ``consistent_if``, ``does_not_protect_against``, ``n``,
        ``method``.
    """
    from ._survey import check_weights

    yv = np.asarray(y, dtype=float).ravel()
    d = np.asarray(D, dtype=float).ravel()
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.shape[0] != yv.size:
        Xm = Xm.T
    if Xm.shape[0] != yv.size:
        raise ValueError("X must have one row per outcome.")
    n = yv.size
    if d.size != n:
        raise ValueError(f"D has {d.size} entries for {n} outcomes.")
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")
    if d.sum() < 2 or (n - d.sum()) < 2:
        raise ValueError("need at least 2 units in each treatment arm.")
    w = np.ones(n) if sampling_weights is None else \
        check_weights(sampling_weights, n, "sampling_weights")
    Z = np.column_stack([np.ones(n), Xm])
    if ps is None:
        b = np.zeros(Z.shape[1])
        for _ in range(50):
            p = 1.0 / (1.0 + np.exp(-Z @ b))
            W = np.maximum(p * (1 - p), 1e-8)
            b = b + np.linalg.pinv((Z * W[:, None]).T @ Z) @ (Z.T @ (d - p))
        e = 1.0 / (1.0 + np.exp(-Z @ b))
    else:
        e = np.asarray(ps, dtype=float).ravel()
        if e.size != n:
            raise ValueError(f"ps has {e.size} entries for {n}.")
    e = np.clip(e, 1e-6, 1 - 1e-6)
    if mu1 is None:
        c1, *_ = np.linalg.lstsq(Z[d == 1.0], yv[d == 1.0], rcond=None)
        m1 = Z @ c1
    else:
        m1 = np.asarray(mu1, dtype=float).ravel()
    if mu0 is None:
        c0, *_ = np.linalg.lstsq(Z[d == 0.0], yv[d == 0.0], rcond=None)
        m0 = Z @ c0
    else:
        m0 = np.asarray(mu0, dtype=float).ravel()
    infl = (m1 - m0 + d * (yv - m1) / e - (1 - d) * (yv - m0) / (1 - e))
    tau = float(np.sum(w * infl) / np.sum(w))
    var = float(np.sum(w ** 2 * (infl - tau) ** 2) / np.sum(w) ** 2)
    return RichResult(payload={
        "ate": tau, "influence": infl, "se": float(np.sqrt(max(var, 0.0))),
        "min_ps": float(e.min()), "max_ps": float(e.max()),
        "consistent_if": "EITHER the propensity model or the outcome model is "
                         "correct -- not both, and not neither",
        "does_not_protect_against": "unmeasured confounding, at all",
        "design_note": "sampling weights multiply the influence function: the "
                       "design and the treatment mechanism are separate randomness",
        "n": int(n),
        "method": "Doubly-robust (AIPW) estimator under a sampling design"})


def cheatsheet():
    return "surdrl: EITHER model right suffices -- two wrong models do not rescue each other"
