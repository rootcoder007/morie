# morie.fn -- function file (rootcoder007/morie)
"""Average treatment effect from a difference-in-differences design."""

from . import _array_core as np

from ._did import add_intercept, ols_fit
from ._richresult import RichResult

__all__ = ["avg_treatment_did"]


def avg_treatment_did(y, D, X=None, y_pre=None, assume="conditional"):
    r"""The ATE, and what it costs to get one out of a DiD design.

    Difference-in-differences identifies the effect **on the treated**.
    It says nothing about untreated units, because their trend is what
    parallel trends assumes, not what it estimates. Turning an
    :math:`ATT` into an :math:`ATE` therefore always requires an extra
    assumption, and this function makes the reader pick it rather than
    hiding it.

    The decomposition is

    .. math:: ATE = ATT \cdot P(D=1) + ATU \cdot P(D=0),

    so :math:`ATT \cdot P(D=1)` is only the TREATED GROUP'S SHARE of
    the ATE -- it is not the ATE, and it is not even a lower bound
    unless :math:`ATU \geq 0`. It is returned as
    ``treated_contribution`` with that name for a reason.

    Two assumptions are supported.

    ``assume='homogeneous'``
        Effects do not vary, so :math:`ATU = ATT` and
        :math:`ATE = ATT`. Honest, and usually the assumption people
        make without saying so.

    ``assume='conditional'`` (default, requires ``X``)
        Effects vary only with the covariates: fit the trend
        :math:`\Delta Y` separately in each arm and set
        :math:`ATE = \mathbb{E}_n[\hat m_1(X) - \hat m_0(X)]` over
        ALL units, which is the regression-imputation estimator. This
        is the extrapolation being made explicit -- the treated model
        is evaluated at control covariate values -- so
        ``extrapolation`` reports how far outside the treated
        covariate range that goes.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome change :math:`\Delta Y`, or the post-period level if
        ``y_pre`` is supplied.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p), optional
        Covariates. Required for ``assume='conditional'``.
    y_pre : array-like, shape (n,), optional
        Pre-period outcome; ``y`` is then the post-period level and
        the difference is taken here.
    assume : {'conditional', 'homogeneous'}
        The assumption used to reach the untreated.

    Returns
    -------
    RichResult
        ``estimate`` (the ATE), ``se``, ``ci``, ``att``, ``atu``,
        ``p_treated``, ``treated_contribution``, ``assumption``,
        ``extrapolation``, ``identity_check``.

    References
    ----------
    Heckman, Ichimura and Todd (1997), *ReStud* 64:605-654.
    Abadie (2005), *Review of Economic Studies* 72:1-19.
    Sloczynski (2022), *Review of Economics and Statistics* 104:501-509.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=4000)
    >>> D = (rng.uniform(size=4000) < 1 / (1 + np.exp(-x))).astype(float)
    >>> dY = 0.5 * x + D * (1.0 + 2.0 * x) + rng.normal(scale=0.3, size=4000)
    >>> out = avg_treatment_did(dY, D, x)
    >>> bool(abs(out["estimate"] - 1.0) < 0.1), bool(out["att"] > 1.2)
    (True, True)
    """
    dY = np.asarray(y, dtype=float).ravel()
    Dv = np.asarray(D, dtype=float).ravel()
    if y_pre is not None:
        dY = dY - np.asarray(y_pre, dtype=float).ravel()
    n = dY.size
    if Dv.size != n:
        raise ValueError(
            "y has %d entries and D has %d." % (n, Dv.size)
        )
    if not np.all(np.isin(Dv, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")
    nt, nc = int(Dv.sum()), int(n - Dv.sum())
    if nt < 2 or nc < 2:
        raise ValueError(
            "need at least 2 treated and 2 control units, got %d and %d."
            % (nt, nc)
        )
    if assume not in ("conditional", "homogeneous"):
        raise ValueError("assume must be 'conditional' or 'homogeneous'.")
    if assume == "conditional" and X is None:
        raise ValueError(
            "assume='conditional' needs covariates: without X there is "
            "nothing to extrapolate the treated effect along, and the only "
            "route to an ATE is assume='homogeneous'."
        )

    p1 = float(Dv.mean())
    tr = Dv == 1
    ct = ~tr
    att_simple = float(dY[tr].mean() - dY[ct].mean())

    if assume == "homogeneous":
        att = att_simple
        atu = att
        ate = att
        v = dY[tr].var(ddof=1) / nt + dY[ct].var(ddof=1) / nc
        se = float(np.sqrt(v))
        extrap = None
    else:
        Xd = add_intercept(np.asarray(X, dtype=float))
        if Xd.shape[0] != n:
            raise ValueError(
                "X has %d rows for %d observations." % (Xd.shape[0], n)
            )
        b1 = ols_fit(Xd[tr], dY[tr])
        b0 = ols_fit(Xd[ct], dY[ct])
        tau_i = Xd @ (b1 - b0)
        ate = float(tau_i.mean())
        att = float(tau_i[tr].mean())
        atu = float(tau_i[ct].mean())
        r = dY - np.where(tr, Xd @ b1, Xd @ b0)
        # influence function of the mean imputed effect: the spread of the
        # per-unit effects, plus each arm's contribution through its own
        # estimated coefficients evaluated at the FULL-sample covariate mean
        xbar = Xd.mean(axis=0)
        A1 = n * (np.linalg.pinv(Xd[tr].T @ Xd[tr]) @ xbar)
        A0 = n * (np.linalg.pinv(Xd[ct].T @ Xd[ct]) @ xbar)
        infl = (tau_i - ate) + np.where(tr, Xd @ A1, -(Xd @ A0)) * r
        se = float(np.sqrt(np.sum(infl**2)) / n)
        lo, hi = Xd[tr].min(axis=0), Xd[tr].max(axis=0)
        outside = np.any((Xd[ct] < lo) | (Xd[ct] > hi), axis=1)
        extrap = {
            "control_rows_outside_treated_support": int(outside.sum()),
            "share": float(outside.mean()),
            "note": (
                "the treated trend model is evaluated at these control "
                "covariate values; that extrapolation is the assumption, "
                "not a technicality"
            ),
        }

    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": ate,
            "se": se,
            "ci": (ate - z * se, ate + z * se),
            "att": att,
            "atu": atu,
            "att_unadjusted": att_simple,
            "p_treated": p1,
            "treated_contribution": att * p1,
            "contribution_note": (
                "ATT * P(D=1) is the treated group's SHARE of the ATE, not "
                "the ATE; the rest is ATU * P(D=0) and DiD does not identify "
                "it without a further assumption"
            ),
            "identity_check": float(ate - (att * p1 + atu * (1 - p1))),
            "assumption": (
                "effects are constant, so ATU = ATT"
                if assume == "homogeneous"
                else "effects vary only with X, so the treated trend model "
                "extrapolates to control covariate values"
            ),
            "extrapolation": extrap,
            "n": int(n),
            "n_treated": nt,
            "n_control": nc,
            "method": "Average treatment effect from a DiD design (%s)" % assume,
        }
    )


def cheatsheet():
    return (
        "avtdid: ATE from a DiD design; DiD identifies ATT, so the "
        "assumption reaching the untreated is explicit and reported"
    )
