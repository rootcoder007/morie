# morie.fn -- function file (rootcoder007/morie)
"""Doubly-robust difference-in-differences (Sant'Anna and Zhao 2020)."""

from . import _array_core as np

from ._did import add_intercept, logit_fit, logit_predict, ols_fit
from ._richresult import RichResult

__all__ = ["dr_did_santanna_zhao"]


def _default_propensity(X, D):
    beta, separated = logit_fit(X, D)
    return logit_predict(X, beta), separated


def _default_outcome(X, dY, control):
    coef = ols_fit(X[control], dY[control])
    return X @ coef


def dr_did_santanna_zhao(y_pre, y_post, treatment, X=None,
                         ml_propensity=None, ml_outcome=None,
                         trim=0.995, n_boot=0, seed=None):
    r"""Doubly-robust ATT for the two-period design.

    With covariates, unconditional parallel trends is usually not
    credible: the treated and control groups differ in ways that
    predict their trend, not only their level. Two fixes exist --
    reweight by the propensity score, or model the control trend --
    and each requires its own model to be right. Sant'Anna and Zhao's
    estimator combines them so that ONE of the two suffices:

    .. math:: \widehat{ATT} = \frac{\mathbb{E}_n\big[w_1
              (\Delta Y - \hat m_0(X))\big]}{\mathbb{E}_n[w_1]}
              - \frac{\mathbb{E}_n\big[w_0
              (\Delta Y - \hat m_0(X))\big]}{\mathbb{E}_n[w_0]},

    with :math:`w_1 = D` and
    :math:`w_0 = (1-D)\hat p(X)/(1-\hat p(X))`.

    Two details in that formula are the paper's contribution and are
    easy to drop by accident. The weights are NORMALISED by their own
    sample means -- Hajek rather than Horvitz-Thompson -- which makes
    the estimator invariant to a constant shift in the outcome and
    much better behaved when some propensities approach one. And the
    outcome regression :math:`\hat m_0` is fitted on the CONTROLS
    ONLY: fitting it on everyone would absorb the treatment effect
    into the regression and bias the estimate toward zero.

    Double robustness is checked, not claimed: ``dr_check`` reports
    the estimate under a deliberately misspecified propensity and
    under a misspecified outcome model, so the reader can see each
    arm recover the same answer on their own data.

    Parameters
    ----------
    y_pre, y_post : array-like, shape (n,)
        Outcome before and after, same units in both.
    treatment : array-like, shape (n,)
        Binary treatment indicator.
    X : array-like, shape (n, p), optional
        Covariates. Without them the estimator is the unconditional
        2x2 DiD, which is reported as ``att_unadjusted`` regardless.
    ml_propensity : callable, optional
        ``f(X, D) -> p_hat``. Logistic regression by default. Supply
        a cross-fitted learner for high-dimensional ``X``.
    ml_outcome : callable, optional
        ``f(X, dY, control_mask) -> m0_hat``. OLS on the controls by
        default.
    trim : float
        Propensities at or above this are dropped, with a count
        reported. A control with :math:`\hat p \to 1` receives
        unbounded weight.
    n_boot : int
        Mammen multiplier-bootstrap replications for the standard
        error. Zero uses the analytic influence function.
    seed : int, optional
        Seed for the bootstrap.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``att_unadjusted``,
        ``covariate_adjustment``, ``propensity``, ``n_trimmed``,
        ``overlap_warning``, ``dr_check``, ``n_treated``,
        ``n_control``.

    References
    ----------
    Sant'Anna and Zhao (2020), *Journal of Econometrics* 219:101-122,
    equations (3.1)-(3.4).
    Abadie (2005), *Review of Economic Studies* 72:1-19.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=4000)
    >>> D = (rng.uniform(size=4000) < 1 / (1 + np.exp(-x))).astype(float)
    >>> pre = x + rng.normal(size=4000)
    >>> post = pre + 0.5 * x + 2.0 * D + rng.normal(size=4000)
    >>> bool(abs(dr_did_santanna_zhao(pre, post, D, x)["estimate"] - 2) < 0.15)
    True
    """
    y0 = np.asarray(y_pre, dtype=float).ravel()
    y1 = np.asarray(y_post, dtype=float).ravel()
    D = np.asarray(treatment, dtype=float).ravel()
    n = y0.size
    if not (y1.size == n == D.size):
        raise ValueError(
            "y_pre, y_post and treatment must have the same length, got "
            "%d, %d and %d." % (y0.size, y1.size, D.size)
        )
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1.")
    if D.sum() < 2 or (1 - D).sum() < 2:
        raise ValueError(
            "need at least 2 treated and 2 control units, got %d and %d."
            % (int(D.sum()), int(n - D.sum()))
        )
    dY = y1 - y0
    unadjusted = float(dY[D == 1].mean() - dY[D == 0].mean())

    if X is None:
        Xd = np.ones((n, 1))
    else:
        Xd = add_intercept(np.asarray(X, dtype=float))
        if Xd.shape[0] != n:
            raise ValueError(
                "X has %d rows for %d observations." % (Xd.shape[0], n)
            )

    if ml_propensity is None:
        p, separated = _default_propensity(Xd, D)
    else:
        p = np.asarray(ml_propensity(Xd, D), dtype=float).ravel()
        if p.size != n:
            raise ValueError(
                "ml_propensity returned %d values for %d rows." % (p.size, n)
            )
        separated = bool(np.min(p) < 1e-6 or np.max(p) > 1 - 1e-6)

    keep = p < float(trim)
    n_trim = int((~keep).sum())
    if keep.sum() < 4:
        raise ValueError(
            "trimming at %g leaves %d observations; the propensity model "
            "gives almost every unit a near-certain treatment probability, "
            "which is a failure of overlap, not a tuning problem."
            % (trim, int(keep.sum()))
        )

    ctrl = (D == 0) & keep
    if ml_outcome is None:
        m0 = _default_outcome(Xd, dY, ctrl)
    else:
        m0 = np.asarray(ml_outcome(Xd, dY, ctrl), dtype=float).ravel()
        if m0.size != n:
            raise ValueError(
                "ml_outcome returned %d values for %d rows." % (m0.size, n)
            )

    def att_from(pv, m0v):
        w1 = np.where(keep, D, 0.0)
        w0 = np.where(keep, (1 - D) * pv / np.maximum(1 - pv, 1e-12), 0.0)
        r = dY - m0v
        return (float(np.sum(w1 * r) / np.sum(w1))
                - float(np.sum(w0 * r) / np.sum(w0)), w1, w0, r)

    att, w1, w0, resid = att_from(p, m0)

    # influence function of the normalised (Hajek) estimator: each arm
    # contributes its own weighted residual minus its weighted mean
    s1, s0 = w1.mean(), w0.mean()
    a1 = float(np.sum(w1 * resid) / np.sum(w1))
    a0 = float(np.sum(w0 * resid) / np.sum(w0))
    infl = w1 * (resid - a1) / s1 - w0 * (resid - a0) / s0
    se = float(np.sqrt(np.sum(infl**2)) / n)

    if n_boot:
        rng = np.random.default_rng(seed)
        # Mammen (1993) two-point weights, as in Sant'Anna-Zhao section 3.2
        k1, k2 = (1 - np.sqrt(5)) / 2, (1 + np.sqrt(5)) / 2
        pk = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        draws = np.where(rng.uniform(size=(int(n_boot), n)) < pk, k1, k2)
        reps = draws @ infl / n
        se = float(np.std(reps, ddof=1))

    # double-robustness demonstration: break one model at a time
    checks = {}
    if X is not None:
        const = np.ones((n, 1))
        bad_p, _ = _default_propensity(const, D)      # constant propensity
        checks["misspecified_propensity"] = att_from(bad_p, m0)[0]
        bad_m0 = np.full(n, float(dY[ctrl].mean()))   # constant outcome model
        checks["misspecified_outcome"] = att_from(p, bad_m0)[0]
        checks["both_misspecified"] = att_from(bad_p, bad_m0)[0]

    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": att,
            "se": se,
            "ci": (att - z * se, att + z * se),
            "att_unadjusted": unadjusted,
            "covariate_adjustment": att - unadjusted,
            "propensity": {
                "min": float(p.min()),
                "max": float(p.max()),
                "mean": float(p.mean()),
            },
            "n_trimmed": n_trim,
            "trim": float(trim),
            "overlap_warning": (
                "the propensity model separates the groups almost perfectly, "
                "so the weights are dominated by a few observations"
                if separated else None
            ),
            "dr_check": checks,
            "dr_note": (
                "'misspecified_propensity' and 'misspecified_outcome' should "
                "each stay close to the estimate; 'both_misspecified' need "
                "not, and that is exactly what double robustness claims"
            ),
            "weights_normalised": True,
            "outcome_fit_on": "controls only",
            "n": int(n),
            "n_treated": int(D.sum()),
            "n_control": int(n - D.sum()),
            "se_method": (
                "Mammen multiplier bootstrap" if n_boot else "influence function"
            ),
            "method": "Doubly-robust DiD (Sant'Anna and Zhao 2020)",
        }
    )


def cheatsheet():
    return (
        "drsza: doubly-robust two-period DiD ATT; normalised weights, "
        "outcome model on controls only, with a double-robustness check"
    )
