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
    Because every model is linear, the plug-in through those models
    reduces to the product of coefficients along the selected paths,
    ``psi = q_D + sum_k q_Mk delta_k`` with
    ``delta_k = path_k a_k + sum_{j<k} c_kj delta_j``, and its influence
    curve is the delta method applied to the stacked least-squares
    influence functions ``(X'X/n)^{-1} x_i e_i``.  No fluctuation is
    applied: the point-treatment clever covariate
    ``D/g - (1 - D)/(1 - g)`` solves the score of the TOTAL effect, and
    fitting it here would move the estimate toward that parameter
    rather than the path-specific one (``eps`` is reported as 0).

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
    p1 = len(W[0])
    asg = [1.0 if pv[k] > 0.5 else 0.0 for k in range(K)]

    def ols_if(des, t):
        """OLS coefficients and their influence functions
        (X'X/n)^{-1} x_i e_i, one row per subject."""
        beta, _, resid, xtxinv = S.ols(des, t)
        q = len(beta)
        inf = [[n * sum(xtxinv[r][c] * des[i][c] for c in range(q)) * resid[i]
                for r in range(q)] for i in range(n)]
        return beta, inf

    # mediator k on (D, 1, X, M_1..M_{k-1}); outcome on (D, 1, X, M)
    mb, mif = [], []
    for k in range(K):
        des = [[Dv[i]] + list(W[i]) + [Mm[i][j] for j in range(k)] for i in range(n)]
        bk, ik = ols_if(des, [Mm[i][k] for i in range(n)])
        mb.append(bk)
        mif.append(ik)
    qdes = [[Dv[i]] + list(W[i]) + list(Mm[i]) for i in range(n)]
    qb, qif = ols_if(qdes, yv)

    # Under the linear models the covariate terms cancel from
    # Q(1, W, M*) - Q(0, W, M0), and the counterfactual mediator gap
    # delta_k = M*_k - M0_k is the same for every subject:
    # delta_k = asg_k a_k + sum_{j<k} c_kj delta_j.
    delta = []
    for k in range(K):
        delta.append(asg[k] * mb[k][0]
                     + sum(mb[k][1 + p1 + j] * delta[j] for j in range(k)))
    psi = qb[0] + sum(qb[1 + p1 + k] * delta[k] for k in range(K))
    # adjoint lam_k = d psi / d delta_k (reverse sweep), then the
    # delta-method influence curve from the stacked OLS influence
    # functions: d psi/d a_k = lam_k asg_k, d psi/d c_kj = lam_k delta_j
    lam = [0.0] * K
    for k in range(K - 1, -1, -1):
        lam[k] = qb[1 + p1 + k] + sum(lam[m_] * mb[m_][1 + p1 + k] for m_ in range(k + 1, K))
    ic = []
    for i in range(n):
        v = qif[i][0] + sum(delta[k] * qif[i][1 + p1 + k] for k in range(K))
        for k in range(K):
            v += lam[k] * asg[k] * mif[k][i][0]
            v += sum(lam[k] * delta[j] * mif[k][i][1 + p1 + j] for j in range(k))
        ic.append(v)
    eps = 0.0
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "n_path": float(sum(pv)), "n": n,
        "method": "Path-specific effect under linear structural models, delta-method influence curve"})


def cheatsheet():
    return "tmlpse: TMLE for a path-specific effect."
