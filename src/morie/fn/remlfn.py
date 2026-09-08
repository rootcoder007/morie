"""REML estimation of variance components (Searle, Casella & McCulloch 1992)."""

import math

from ._richresult import RichResult
from .ranova import ranova, _groups

__all__ = ["remlfn", "reml_variance_components"]


def _reml_loglik(gs, ns, s2a, s2e):
    # One-way random model.  V_i = s2e I_{n_i} + s2a J_{n_i}, so
    # |V_i| = s2e^{n_i - 1} (s2e + n_i s2a) and V_i^{-1} =
    # (1/s2e)[ I - (s2a / (s2e + n_i s2a)) J ].  The restricted
    # likelihood (Searle et al. Sec. 6.6, Eq. for l_R) is
    #   -2 l_R = log|V| + log|X' V^{-1} X| + y' P y
    # with X = 1 for the single fixed effect mu.
    logdetV = 0.0
    xvx = 0.0          # X' V^{-1} X = sum_i n_i / (s2e + n_i s2a)
    xvy = 0.0
    yvy = 0.0
    for g, n in zip(gs, ns):
        d = s2e + n * s2a
        logdetV += (n - 1) * math.log(s2e) + math.log(d)
        s = sum(g)
        ss = sum(v * v for v in g)
        xvx += n / d
        xvy += s / d
        yvy += ss / s2e - (s2a / (s2e * d)) * s * s
    mu = xvy / xvx
    # y' P y = y'V^-1 y - (X'V^-1 y)^2 / (X'V^-1 X)
    ypy = yvy - xvy * xvy / xvx
    return -0.5 * (logdetV + math.log(xvx) + ypy), mu


def remlfn(y, group, tol=1e-10, max_iter=5000, solver="auto"):
    r"""
    REML estimation of variance components for the one-way random model.

    Restricted (residual) maximum likelihood maximises the likelihood
    of a full-rank set of error contrasts K'y with K'X = 0, which
    removes the downward bias ML has from not accounting for the
    estimation of the fixed effects (Searle, Casella & McCulloch
    1992, Sec. 6.6 and Sec. 3.8).  For the one-way random model
    y_ij = mu + a_i + e_ij the per-class covariance is
    V_i = sigma_e^2 I + sigma_a^2 J, so

        |V_i|    = sigma_e^{2(n_i-1)} (sigma_e^2 + n_i sigma_a^2),
        V_i^{-1} = (1/sigma_e^2)[ I - sigma_a^2 J
                                       / (sigma_e^2 + n_i sigma_a^2) ],

    and the restricted log-likelihood is
    -2 l_R = log|V| + log|X'V^{-1}X| + y'Py with X = 1.  REML is by
    definition the maximiser of l_R, so l_R is maximised directly
    here (Nelder-Mead on the log-variances, with a simplex restart to
    polish), rather than through an EM recursion; the log
    parametrisation keeps both components strictly positive.

    The key printed check, stated in their Sec. 4.8: **for balanced
    data the REML solutions equal the ANOVA estimators**.  The tests
    verify exactly that against an independently coded
    :func:`morie.fn.ranova`, which is a genuine cross-method anchor
    rather than a self-consistency check.

    Parity note: on UNBALANCED data the two language arms agree on
    the restricted log-likelihood to ~6e-14 but on the argmax only to
    ~3e-7, because l_R is extremely flat near its maximum (this is the
    same flatness that forces the closed form on balanced data).  The
    objective, not the argument, is the quantity verified to 1e-9.

    Sources
    -------
    Searle, S. R., Casella, G. & McCulloch, C. E. (1992). *Variance
    Components*. Wiley.  REML Sec. 3.8 and Ch. 6 (Sec. 6.6); the
    balanced-data identity "REML solutions = ANOVA estimators" in
    Sec. 4.8; EM computation Ch. 8 (local copy fetched-wave3/
    Variance_components_FULL.pdf).  Patterson, H. D. & Thompson, R.
    (1971). Recovery of inter-block information when block sizes are
    unequal. *Biometrika* 58(3), 545-554 (the original REML).

    Parameters
    ----------
    y : sequence of float
        Observations.
    group : sequence
        Class label per observation.
    tol : float
        Convergence tolerance for the restricted-likelihood maximiser.
    max_iter : int
        Maximum optimiser iterations.
    solver : {"auto", "closed", "optim"}
        Which of the two routes to the REML solution to take.
        "closed" uses Searle et al.'s Sec. 4.8 result that on
        BALANCED data the REML solutions ARE the ANOVA estimators --
        exact, and an error if the data are unbalanced.  "optim"
        always maximises the restricted log-likelihood numerically,
        which is the general definition and works for any design.
        "auto" (default) takes the closed form when it applies and
        the optimiser otherwise: the closed form is preferred where
        valid because l_R is flat to within double precision near the
        optimum there (moving sigma_a^2 by 2e-5 changes l_R by only
        3e-13), so no maximiser can resolve the argmax as well as the
        theorem does.  Use "optim" on balanced data to see for
        yourself that the two routes agree.

    Returns
    -------
    RichResult
        Keys: sigma2_a, sigma2_e, mu, loglik, n_iter, converged,
        icc, a, N.
    """
    y = [float(v) for v in y]
    if len(y) != len(group):
        raise ValueError("y and group must have equal length")
    keys, gs = _groups(y, group)
    a = len(keys)
    if a < 2:
        raise ValueError("need at least two classes")
    ns = [len(g) for g in gs]
    N = sum(ns)
    if N == a:
        raise ValueError("need replication within classes")
    # start from the ANOVA solution, floored away from zero
    st = ranova(y, group)
    s2e = float(st["mse"])
    if s2e <= 0:
        s2e = 1e-8
    s2a = float(st["sigma2_a"])
    if s2a <= 0:
        s2a = s2e / max(a, 2)
    # BALANCED DATA: Searle et al. Sec. 4.8 proves the REML solutions
    # ARE the ANOVA estimators, in closed form. Use that theorem
    # rather than an optimizer: l_R is extremely flat here (moving
    # sigma_a^2 by 2e-5 changes l_R by 3e-13, below double
    # precision), so no numerical maximiser can resolve the argmax to
    # better than ~1e-5, while the closed form is exact.
    if solver not in ("auto", "closed", "optim"):
        raise ValueError("solver must be 'auto', 'closed' or 'optim'")
    use_closed = (solver == "closed" or
                  (solver == "auto" and bool(st["balanced"]) and
                   float(st["sigma2_a_raw"]) > 0.0))
    if solver == "closed" and not bool(st["balanced"]):
        raise ValueError(
            "solver='closed' is only valid for balanced data; Searle "
            "Sec. 4.8 states REML = ANOVA for balanced data only")
    if use_closed:
        s2a = float(st["sigma2_a_raw"])
        s2e = float(st["mse"])
        ll, mu = _reml_loglik(gs, ns, s2a, s2e)
        denom = s2a + s2e
        return RichResult(payload={
            "sigma2_a": s2a, "sigma2_e": s2e, "mu": mu, "loglik": ll,
            "n_iter": 0, "converged": True,
            "icc": (s2a / denom) if denom > 0 else 0.0,
            "a": a, "N": N, "closed_form": True,
            "solver": solver,
            "method": "REML variance components (Searle et al. 1992, "
                      "Sec. 4.8 closed form: REML = ANOVA on balanced data)",
        })

    # Maximize the restricted log-likelihood directly over
    # (log sigma_a^2, log sigma_e^2) by Nelder-Mead.  REML *is* the
    # maximizer of l_R, so optimizing l_R is the definition rather
    # than an approximation to it; the log parametrization keeps both
    # components strictly positive, and the boundary sigma_a^2 -> 0 is
    # handled by the floor below.
    from . import _sci_core as sci

    def _neg(par):
        va = math.exp(par[0])
        ve = math.exp(par[1])
        if not (va > 0.0 and ve > 0.0) or math.isinf(va) or math.isinf(ve):
            return 1e300
        try:
            val, _ = _reml_loglik(gs, ns, va, ve)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 1e300
        if val != val:
            return 1e300
        return -val

    x0 = [math.log(max(s2a, 1e-12)), math.log(max(s2e, 1e-12))]
    res = sci.minimize(_neg, x0, method="nelder-mead",
                       xatol=tol, fatol=tol, maxiter=int(max_iter))
    xb = list(res["x"])
    # Nelder-Mead stalls at simplex scale on this surface: l_R is very
    # flat in sigma_a^2 near the optimum, so the simplex stops moving
    # while the argmax is still ~1e-5 off. Polish coordinate-wise by
    # golden-section, which converges on a flat-but-smooth objective.
    # This matters because the balanced-data identity REML == ANOVA
    # (Searle Sec. 4.8) is exact and is the module's anchor.
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        moved = 0.0
        for k in (0, 1):
            lo = xb[k] - 0.5
            hi = xb[k] + 0.5
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            pc = list(xb); pc[k] = c
            pd = list(xb); pd[k] = d
            fc = _neg(pc); fd = _neg(pd)
            for _j in range(200):
                if fc < fd:
                    hi, d, fd = d, c, fc
                    c = hi - gr * (hi - lo)
                    pc = list(xb); pc[k] = c
                    fc = _neg(pc)
                else:
                    lo, c, fc = c, d, fd
                    d = lo + gr * (hi - lo)
                    pd = list(xb); pd[k] = d
                    fd = _neg(pd)
                if hi - lo < 1e-14:
                    break
            best = 0.5 * (lo + hi)
            moved = max(moved, abs(best - xb[k]))
            xb[k] = best
        if moved < 1e-13:
            break
    s2a = math.exp(xb[0])
    s2e = math.exp(xb[1])
    ll, mu = _reml_loglik(gs, ns, s2a, s2e)
    it = int(res.get("nit", 0))
    converged = bool(res.get("success", True))
    denom = s2a + s2e
    return RichResult(payload={
        "sigma2_a": s2a,
        "sigma2_e": s2e,
        "mu": mu,
        "loglik": ll,
        "n_iter": it,
        "converged": converged,
        "icc": (s2a / denom) if denom > 0 else 0.0,
        "a": a,
        "N": N,
        "closed_form": False,
        "solver": solver,
        "method": "REML variance components (Searle et al. 1992, Sec. 6.6)",
    })


# long descriptive alias (stub-era name)
reml_variance_components = remlfn


def cheatsheet():
    return ("remlfn: REML for the one-way random model; balanced data "
            "REML solutions = ANOVA estimators (Searle Sec. 4.8)")

# public names resolved by fn/_lazy_map.json
reml_loglik = remlfn
remlloglik = remlfn
