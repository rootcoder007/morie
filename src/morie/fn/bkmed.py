# morie.fn -- function file (rootcoder007/morie)
"""Baron-Kenny stepwise mediation."""

from __future__ import annotations

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["baron_kenny"]


def _ols(design, y):
    """OLS with an intercept prepended; returns coefficients and their SEs."""
    D = np.column_stack([np.ones(design.shape[0]), design])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    dof = y.size - D.shape[1]
    if dof <= 0:
        raise ValueError(f"Not enough observations for {D.shape[1]} parameters; got {y.size}.")
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(D.T @ D)
    se = np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 0.0))
    return beta, se, dof


def baron_kenny(Y, X, M, alpha=0.05):
    r"""Baron & Kenny's stepwise test for mediation.

    Three regressions:

    .. math::

        \text{(1)}\; Y &= i_1 + c\,X + e_1 \\
        \text{(2)}\; M &= i_2 + a\,X + e_2 \\
        \text{(3)}\; Y &= i_3 + c'X + b\,M + e_3

    Their four conditions are that :math:`c`, :math:`a` and :math:`b` are
    each significant, and that :math:`|c'| < |c|` -- complete mediation
    when :math:`c'` is no longer distinguishable from zero, partial
    mediation when it shrinks but survives.

    Two cautions are reported alongside the verdict rather than left for
    the reader to remember.

    The requirement that :math:`c` be significant is now generally
    regarded as too strong. Mediation can be real when the total effect
    is null, because an indirect path and a direct path of opposite sign
    can cancel; insisting on step 1 discards exactly those cases. The
    ``steps`` field records each condition separately so a caller can
    apply a weaker rule.

    The stepwise logic is also not a test of the indirect effect itself.
    It chains three separate decisions, so its error rate is not the
    nominal one, and it gives no interval for :math:`ab`. Use
    :func:`morie.fn.abind.ab_indirect_effect` for the product with a
    Sobel standard error, or a bootstrap, when the indirect effect is
    the quantity of interest.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        Treatment or predictor, one-dimensional.
    M : array-like, shape (n,)
        Mediator.
    alpha : float, default 0.05
        Significance level for each step.

    Returns
    -------
    RichResult
        keys: ``c`` (total), ``a``, ``b``, ``c_prime`` (direct),
        ``indirect`` (ab), ``proportion_mediated``, ``se``, ``p``
        (per path), ``steps`` (per condition), ``mediation``
        ("complete", "partial" or "none"), ``n``, ``method``.

    References
    ----------
    Baron, R. M. & Kenny, D. A. (1986). The moderator-mediator variable
    distinction in social psychological research: conceptual, strategic,
    and statistical considerations. *Journal of Personality and Social
    Psychology*, 51(6), 1173-1182.
    """
    y = np.asarray(Y, dtype=float).ravel()
    x = np.asarray(X, dtype=float)
    m = np.asarray(M, dtype=float).ravel()
    if x.ndim != 1:
        raise ValueError(f"X must be one-dimensional; got shape {x.shape}.")
    n = y.size
    if not (x.size == n and m.size == n):
        raise ValueError(f"Y, X and M must be the same length; got {n}, {x.size}, {m.size}.")
    if n < 4:
        raise ValueError(f"Need at least 4 observations for the three-step fit, got {n}.")
    if not all(np.all(np.isfinite(v)) for v in (y, x, m)):
        raise ValueError("Y, X and M must be finite.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    b1, s1, d1 = _ols(x.reshape(-1, 1), y)  # step 1: Y ~ X
    b2, s2, d2 = _ols(x.reshape(-1, 1), m)  # step 2: M ~ X
    b3, s3, d3 = _ols(np.column_stack([x, m]), y)  # step 3: Y ~ X + M

    c, se_c = float(b1[1]), float(s1[1])
    a, se_a = float(b2[1]), float(s2[1])
    cp, se_cp = float(b3[1]), float(s3[1])
    b, se_b = float(b3[2]), float(s3[2])

    def pval(coef, se, dof):
        return float(2 * stats.t.sf(abs(coef / se), dof)) if se > 0 else np.nan

    p_c, p_a, p_b, p_cp = (
        pval(c, se_c, d1),
        pval(a, se_a, d2),
        pval(b, se_b, d3),
        pval(cp, se_cp, d3),
    )

    steps = {
        "step1_total_effect_significant": bool(p_c < alpha),
        "step2_x_predicts_m": bool(p_a < alpha),
        "step3_m_predicts_y_given_x": bool(p_b < alpha),
        "step4_direct_effect_shrinks": bool(abs(cp) < abs(c)),
    }
    if steps["step2_x_predicts_m"] and steps["step3_m_predicts_y_given_x"]:
        mediation = "complete" if p_cp >= alpha else ("partial" if steps["step4_direct_effect_shrinks"] else "none")
    else:
        mediation = "none"

    return RichResult(
        title="Baron-Kenny stepwise mediation",
        payload={
            "c": c,
            "a": a,
            "b": b,
            "c_prime": cp,
            "indirect": a * b,
            "proportion_mediated": float(a * b / c) if c != 0 else np.nan,
            "se": {"c": se_c, "a": se_a, "b": se_b, "c_prime": se_cp},
            "p": {"c": p_c, "a": p_a, "b": p_b, "c_prime": p_cp},
            "steps": steps,
            "mediation": mediation,
            "n": int(n),
            "alpha": float(alpha),
            "method": "Baron & Kenny (1986) stepwise mediation",
        },
    )


def cheatsheet():
    return "bkmed: Baron-Kenny stepwise mediation"
