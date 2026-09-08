# morie.fn -- function file (rootcoder007/morie)
r"""Survey-weighted Cox regression, and why the variance is the hard
part.

**The point estimate is easy.** Weight each subject's contribution to
the partial likelihood by its sampling weight. The score becomes

.. math:: U(\beta) = \sum_i w_i \delta_i
          \Bigl\{ x_i - \frac{\sum_{j \in R_i} w_j x_j
          e^{x_j'\beta}}{\sum_{j \in R_i} w_j e^{x_j'\beta}}
          \Bigr\}

and Newton-Raphson on that gives :math:`\hat\beta`. This estimates
the coefficient of the Cox model that would be fitted to the whole
finite population -- a *census parameter*, which is what a survey
analyst usually wants, and which is well defined whether or not the
proportional hazards model is true.

**The variance is not.** The inverse information :math:`I^{-1}` is
the variance of a maximum likelihood estimator under independent
sampling from the model. Under a survey design the observations are
neither independent nor identically weighted, and :math:`I^{-1}`
understates the variance whenever the design clusters and overstates
it when it stratifies on something predictive. Binder's answer is a
sandwich built from the *design*:

.. math:: \hat{V}(\hat\beta) = I^{-1}
          \, \hat{V}_{\text{design}}\Bigl(\sum_i w_i \hat{U}_i\Bigr)
          \, I^{-1}

where :math:`\hat{U}_i` is subject :math:`i`'s score residual (its
own contribution to :math:`U` plus the contribution it makes to the
risk sets of every earlier failure). The middle term is the
design-based variance of a total -- ordinary stratified,
between-cluster variance -- so the survey machinery is applied to the
residuals rather than to the coefficient.

**What that buys, concretely.** Clustering inflates the variance and
the model-based standard error does not see it. The anchor builds a
design where subjects within a cluster share their covariate, and
shows Binder's standard error rising with the intra-cluster
correlation while the model-based one does not move.

**Ties.** Breslow's approximation, which is what weighting the
partial likelihood naturally yields: each tied failure sees the full
risk set. :mod:`morie.fn.coxph` uses Efron's approximation instead,
so the two agree exactly when failure times are distinct and diverge
slightly when they are not. The anchor exhibits both facts rather
than only the agreeable one.

**Two checks that must hold exactly.** With every weight equal to one
and no design structure, :math:`\hat\beta` must equal the unweighted
Cox estimate; and integer weights must give the same
:math:`\hat\beta` as physically replicating those rows. Both are
identities, and both are anchored.

References
----------
Binder, D. A. (1992) "Fitting Cox's proportional hazards models from
survey data", *Biometrika* 79(1), 139-147,
doi:10.1093/biomet/79.1.139. The weighted estimating equation above,
the interpretation of :math:`\hat\beta` as a finite-population census
parameter, and the sandwich variance with the design-based variance
of the weighted score residuals in the middle.

Cox, D. R. (1972) "Regression models and life-tables", *Journal of
the Royal Statistical Society: Series B* 34(2), 187-202,
doi:10.1111/j.2517-6161.1972.tb00899.x, for the partial likelihood
being weighted; see :mod:`morie.fn.coxph` for the unweighted fit this
must reduce to.
"""

import math

from ._richresult import RichResult

__all__ = ["svycoxph", "score_residuals", "survey_cox"]


def _prep(time, event, X, weights, strata, cluster):
    T = [float(t) for t in time]
    E = [int(e) for e in event]
    n = len(T)
    if len(E) != n:
        raise ValueError("svycox: %d times but %d event indicators"
                         % (n, len(E)))
    if n < 2:
        raise ValueError("svycox: need at least two subjects")
    if any(e not in (0, 1) for e in E):
        raise ValueError("svycox: the event indicator must be 0 or 1")
    if any(t < 0 for t in T):
        raise ValueError("svycox: a survival time cannot be negative")
    M = [[float(v) for v in row] for row in X]
    if len(M) != n:
        raise ValueError("svycox: %d covariate rows but %d subjects"
                         % (len(M), n))
    p = len(M[0])
    if p == 0 or any(len(r) != p for r in M):
        raise ValueError("svycox: the covariate matrix is ragged or "
                         "empty")
    w = [1.0] * n if weights is None else [float(x) for x in weights]
    if len(w) != n:
        raise ValueError("svycox: %d weights but %d subjects"
                         % (len(w), n))
    if any(x <= 0 for x in w):
        raise ValueError("svycox: sampling weights must be positive")
    if not any(E):
        raise ValueError("svycox: no events, so nothing is estimable")
    h = ["1"] * n if strata is None else [str(x) for x in strata]
    c = ([str(i) for i in range(n)] if cluster is None
         else [str(x) for x in cluster])
    if len(h) != n or len(c) != n:
        raise ValueError("svycox: strata and cluster need one entry "
                         "per subject")
    return T, E, M, w, h, c, n, p


def _score_and_info(T, E, M, w, beta, n, p):
    """Weighted score, information, and per-subject score residuals."""
    eta = [sum(M[i][k] * beta[k] for k in range(p)) for i in range(n)]
    r = [w[i] * math.exp(eta[i]) for i in range(n)]
    U = [0.0] * p
    I = [[0.0] * p for _ in range(p)]
    resid = [[0.0] * p for _ in range(n)]
    order = sorted(range(n), key=lambda i: T[i])
    for i in range(n):
        if not E[i]:
            continue
        risk = [j for j in order if T[j] >= T[i]]
        s0 = sum(r[j] for j in risk)
        if s0 <= 0:
            continue
        s1 = [sum(r[j] * M[j][k] for j in risk) for k in range(p)]
        xbar = [s1[k] / s0 for k in range(p)]
        for k in range(p):
            U[k] += w[i] * (M[i][k] - xbar[k])
            resid[i][k] += M[i][k] - xbar[k]
        for k in range(p):
            for l in range(p):
                s2 = sum(r[j] * M[j][k] * M[j][l] for j in risk)
                I[k][l] += w[i] * (s2 / s0 - xbar[k] * xbar[l])
        # every member of the risk set is pulled towards the mean
        for j in risk:
            f = w[i] * r[j] / s0
            for k in range(p):
                resid[j][k] -= f * (M[j][k] - xbar[k]) / max(w[j],
                                                             1e-300)
    return U, I, resid


def _solve(A, b):
    p = len(b)
    Ab = [list(A[i]) + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(Ab[r][c]))
        if abs(Ab[piv][c]) < 1e-14:
            raise ValueError(
                "svycox: the information matrix is singular. Either "
                "a covariate is constant or collinear among the "
                "failures, or the groups are completely separated -- "
                "one always failing before the other -- in which case "
                "the partial likelihood is monotone and no finite "
                "estimate exists")
        Ab[c], Ab[piv] = Ab[piv], Ab[c]
        for r in range(p):
            if r == c:
                continue
            f = Ab[r][c] / Ab[c][c]
            for k in range(c, p + 1):
                Ab[r][k] -= f * Ab[c][k]
    return [Ab[i][p] / Ab[i][i] for i in range(p)]


def _inverse(A):
    p = len(A)
    out = [[0.0] * p for _ in range(p)]
    for j in range(p):
        e = [1.0 if i == j else 0.0 for i in range(p)]
        col = _solve(A, e)
        for i in range(p):
            out[i][j] = col[i]
    return out


def score_residuals(time, event, X, beta, weights=None):
    r"""Per-subject score residuals at ``beta``.

    Their weighted total is the score, which is the property the
    sandwich needs and the anchor checks.
    """
    T, E, M, w, _h, _c, n, p = _prep(time, event, X, weights, None,
                                     None)
    return _score_and_info(T, E, M, w, [float(b) for b in beta],
                           n, p)[2]


def _design_variance(contrib, w, h, c, p):
    """Between-cluster, within-stratum variance of a weighted total."""
    cells = {}
    for i in range(len(w)):
        key = (h[i], c[i])
        if key not in cells:
            cells[key] = [0.0] * p
        for k in range(p):
            cells[key][k] += w[i] * contrib[i][k]
    by_h = {}
    for (hh, _cc), v in cells.items():
        by_h.setdefault(hh, []).append(v)
    V = [[0.0] * p for _ in range(p)]
    for hh, vs in by_h.items():
        nh = len(vs)
        if nh < 2:
            raise ValueError("svycox: stratum %r has a single "
                             "cluster, so its variance contribution "
                             "is not estimable" % hh)
        mean = [sum(v[k] for v in vs) / nh for k in range(p)]
        f = nh / float(nh - 1)
        for v in vs:
            for k in range(p):
                for l in range(p):
                    V[k][l] += f * (v[k] - mean[k]) * (v[l] - mean[l])
    return V


def svycoxph(time, event, X, weights=None, strata=None, cluster=None,
             max_iter=100, tol=1e-9):
    r"""Fit a Cox model to survey data with Binder's variance."""
    T, E, M, w, h, c, n, p = _prep(time, event, X, weights, strata,
                                   cluster)
    beta = [0.0] * p
    hist = []
    for _ in range(int(max_iter)):
        U, I, _ = _score_and_info(T, E, M, w, beta, n, p)
        step = _solve(I, U)
        beta = [beta[k] + step[k] for k in range(p)]
        hist.append(max(abs(s) for s in step))
        if hist[-1] < float(tol):
            break
    else:
        raise ValueError("svycox: Newton-Raphson did not converge in "
                         "%d iterations (last step %.3g)"
                         % (int(max_iter), hist[-1]))
    U, I, resid = _score_and_info(T, E, M, w, beta, n, p)
    Iinv = _inverse(I)
    Vd = _design_variance(resid, w, h, c, p)
    V = [[sum(Iinv[a][k] * Vd[k][l] * Iinv[l][b]
              for k in range(p) for l in range(p))
          for b in range(p)] for a in range(p)]
    se = [math.sqrt(V[k][k]) if V[k][k] > 0 else float("nan")
          for k in range(p)]
    se_model = [math.sqrt(Iinv[k][k]) if Iinv[k][k] > 0
                else float("nan") for k in range(p)]
    return RichResult(payload={
        "estimate": beta, "coefficients": beta,
        "hazard_ratios": [math.exp(b) for b in beta],
        "std_errors": se, "model_std_errors": se_model,
        "vcov": V, "information": I, "score": U,
        "design_effect": [((se[k] / se_model[k]) ** 2
                           if se_model[k] > 0 else float("nan"))
                          for k in range(p)],
        "z": [beta[k] / se[k] if se[k] > 0 else float("nan")
              for k in range(p)],
        "n": n, "n_events": sum(E), "n_iterations": len(hist),
        "ties": "breslow",
        "method": "Binder (1992) weighted partial likelihood with a "
                  "design-based sandwich variance",
    })


def survey_cox(time, event, X, weights=None, strata=None,
               cluster=None, **kw):
    r"""Entry point: see :func:`svycoxph`."""
    return svycoxph(time, event, X, weights, strata, cluster, **kw)


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
surveycox = survey_cox
