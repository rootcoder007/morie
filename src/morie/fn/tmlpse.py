# morie.fn -- function file (rootcoder007/morie)
"""TMLE for a path-specific effect through a chosen subset of mediators."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_path_specific"]


def tmle_path_specific(y, D, M_chain, X, path):
    """Targeted effect of treatment travelling only along the chosen paths.

    A path-specific effect sets the treatment to different values on
    different edges: mediators marked in ``path`` see ``A = 1``, all
    other mediators see ``A = 0``, and the outcome node sees ``A = 1``.
    The contrast is against the all-zero regime.  This is only
    identified when there is no recanting witness -- no mediator that is
    both on and off the selected paths -- which the caller asserts by
    supplying ``path``; the function does not and cannot check the graph
    for it.

    Mediators are taken in the given column order as a causal chain,
    each modelled linearly on treatment, covariates, and the mediators
    before it.  The counterfactual mediator values are generated
    recursively from those models at the path-assigned treatment values,
    so an upstream mediator's counterfactual feeds the downstream one.
    The outcome regression is targeted first, with the point-treatment
    clever covariate ``H = D/g - (1 - D)/(1 - g)``, and the path-specific
    mean is the plug-in through the targeted ``Q``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    M_chain : array-like, shape (n, K)
        Mediators in causal order.
    X : array-like, shape (n, p)
        Baseline covariates.
    path : array-like, shape (K,)
        1 if the treatment is allowed to act through that mediator.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_path``, ``n``.

    References
    ----------
    Miles, C. H., Shpitser, I., Kanki, P., Meloni, S. & Tchetgen
    Tchetgen, E. J. (2017).  Quantifying an adherence path-specific
    effect of antiretroviral therapy in the Nigeria PEPFAR program.
    Journal of the American Statistical Association 112(520):1443-1452.
    doi:10.1080/01621459.2017.1295862.  The no-recanting-witness
    condition is Avin, C., Shpitser, I. & Pearl, J. (2005),
    Identifiability of path-specific effects, IJCAI-05, 357-363.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    pv = C.vec(path)
    n = len(yv)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_path_specific: y and D must share one length")
    Mm = C.mat(M_chain)
    Xm = C.mat(X)
    if len(Mm) != n or len(Xm) != n:
        raise ValueError("tmle_path_specific: M_chain and X must have one row per subject")
    K = len(Mm[0])
    if len(pv) != K:
        raise ValueError("tmle_path_specific: path must have one entry per mediator")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]

    mb = []
    for k in range(K):
        des = [[Dv[i]] + list(W[i]) + [Mm[i][j] for j in range(k)] for i in range(n)]
        b, _, _, _ = S.ols(des, [Mm[i][k] for i in range(n)])
        mb.append(b)

    def gen(assign):
        """Counterfactual mediator chain under per-mediator treatment values."""
        out = [[0.0] * K for _ in range(n)]
        for k in range(K):
            for i in range(n):
                row = [assign[k]] + list(W[i]) + [out[i][j] for j in range(k)]
                out[i][k] = C.dot(row, mb[k])
        return out

    Mstar = gen([1.0 if pv[k] > 0.5 else 0.0 for k in range(K)])
    Mnull = gen([0.0] * K)

    qdes = [[Dv[i]] + list(W[i]) + list(Mm[i]) for i in range(n)]
    qb, _, _, _ = S.ols(qdes, yv)
    Qobs = [C.dot(qdes[i], qb) for i in range(n)]
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1 = [C.dot([1.0] + list(W[i]) + list(Mstar[i]), qb) + eps / g[i] for i in range(n)]
    Q0 = [C.dot([0.0] + list(W[i]) + list(Mnull[i]), qb) - eps / (1.0 - g[i])
          for i in range(n)]
    psi = sum(Q1[i] - Q0[i] for i in range(n)) / n
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Q1[i] - Q0[i] - psi for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "n_path": float(sum(pv)), "n": n,
        "method": "TMLE for a path-specific effect through a chosen mediator subset"})


def cheatsheet():
    return "tmlpse: TMLE for a path-specific effect."
