# morie.fn -- function file (rootcoder007/morie)
"""Outcome-model diagnostic for a marginal structural model."""

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["outcome_model_diagnostic"]


def outcome_model_diagnostic(y, A, H, Q=None):
    """Residual diagnostic and overlap check for an MSM outcome model.

    A marginal structural model is only as good as the outcome model
    behind it, and the failure it hides is systematic: if ``Q`` is
    misspecified in a way that correlates with treatment, the residuals
    carry treatment signal that the model should have absorbed.  The
    diagnostic regresses the standardized residuals on the treatment,

        r_i = (y_i - Q_i) / sd(y - Q),     r ~ b0 + b1 A,

    so ``b1`` near zero with a small t-statistic is the pass condition:
    ``b1`` measures exactly the part of the outcome that treatment still
    explains after the model has had its turn.

    The second failure mode is not the outcome model at all but the
    design: without overlap in the propensity there is no data at some
    treatment levels and any weight is an extrapolation.  Crump et al.
    give a rule for the subsample worth analysing, ``alpha <= e(x) <=
    1 - alpha``, with ``alpha`` the smallest value satisfying

        1 / (alpha (1 - alpha)) <= 2 E[ 1 / (e(1-e)) ;
                                        alpha <= e <= 1 - alpha ],

    solved here by a deterministic scan over the observed propensity
    values.  The propensity is fitted by logistic regression of ``A``
    on ``H``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed outcome.
    A : array-like, shape (n,)
        Binary treatment, 0/1.
    H : array-like, shape (n, p)
        Covariate history, no intercept column; one is added.
    Q : array-like or None
        Fitted values of the outcome model.  If ``None``, the outcome
        is regressed on ``[1, A, H]`` by least squares and its fitted
        values are used.

    Returns
    -------
    RichResult
        ``estimate`` (``b1``), ``b1``, ``t_stat``, ``resid_sd``,
        ``mean_resid_treated``, ``mean_resid_control``, ``alpha_crump``,
        ``n_kept``, ``min_ps``, ``max_ps``, ``n``.

    References
    ----------
    Crump, R. K., Hotz, V. J., Imbens, G. W. & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment
    effects.  Biometrika, 96(1), 187--199.  doi:10.1093/biomet/asn055
    Robins, J. M., Hernan, M. A. & Brumback, B. (2000).  Marginal
    structural models and causal inference in epidemiology.
    Epidemiology, 11(5), 550--560.
    """
    yv = C.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("outcome_model_diagnostic: y is empty")
    Av = C.vec(A)
    if len(Av) != n:
        raise ValueError("outcome_model_diagnostic: y and A have different lengths")
    if any(v not in (0.0, 1.0) for v in Av):
        raise ValueError("outcome_model_diagnostic: A must be binary 0/1")
    Hm = C.mat(H)
    if len(Hm) != n:
        raise ValueError("outcome_model_diagnostic: H and y have different lengths")
    if Q is None:
        des = [[1.0, Av[i]] + list(Hm[i]) for i in range(n)]
        _, fitted, _, _ = C.lstsq(des, yv)
        Qv = fitted
    else:
        Qv = C.vec(Q)
        if len(Qv) != n:
            raise ValueError("outcome_model_diagnostic: Q and y have different lengths")
    e = [yv[i] - Qv[i] for i in range(n)]
    sd = C.sd(e, ddof=1)
    if sd <= 0.0:
        raise ValueError("outcome_model_diagnostic: residuals have zero spread")
    r = [v / sd for v in e]
    beta, fit2, res2, xtxinv = C.lstsq([[1.0, Av[i]] for i in range(n)], r)
    s2 = sum(v * v for v in res2) / (n - 2) if n > 2 else float("nan")
    seb = math.sqrt(s2 * xtxinv[1][1]) if n > 2 and xtxinv[1][1] > 0 else float("nan")
    tstat = beta[1] / seb if seb == seb and seb > 0 else float("nan")
    nt = sum(1 for v in Av if v == 1.0)
    mrt = sum(r[i] for i in range(n) if Av[i] == 1.0) / nt if nt else float("nan")
    mrc = (sum(r[i] for i in range(n) if Av[i] == 0.0) / (n - nt)
           if n - nt else float("nan"))

    Hd = C.cbind1(Hm)
    gam = core.logit_irls(Hd, Av)
    ps = [core.sigmoid(sum(Hd[i][k] * gam[k] for k in range(len(gam))))
          for i in range(n)]
    cand = sorted({min(p, 1.0 - p) for p in ps})
    alpha = 0.0
    for a in cand:
        if a >= 0.5:
            continue
        keep = [p for p in ps if a <= p <= 1.0 - a]
        if not keep:
            continue
        rhs = 2.0 * sum(1.0 / (p * (1.0 - p)) for p in keep) / len(keep)
        if 1.0 / (a * (1.0 - a)) <= rhs:
            alpha = a
            break
    nkeep = sum(1 for p in ps if alpha <= p <= 1.0 - alpha)
    return RichResult(payload={
        "estimate": beta[1], "b0": beta[0], "b1": beta[1], "t_stat": tstat,
        "resid_sd": sd, "mean_resid_treated": mrt,
        "mean_resid_control": mrc, "alpha_crump": alpha,
        "n_kept": float(nkeep), "min_ps": min(ps), "max_ps": max(ps),
        "n": n,
        "method": "MSM outcome-model residual diagnostic with Crump overlap"})


def cheatsheet():
    return "ocmtmd: MSM outcome-model residual diagnostic and overlap check"


outcomemodeldiagnostic = outcome_model_diagnostic
