# morie.fn -- slice s04 (rootcoder007/morie)
"""BayesR: mixture of normals prior with different variance classes.

NOT IN THE BOOK.  Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, was searched in full -- all seventeen page-range
volumes and the index, [Pages 683-691].  Chapter 6, volume [Pages
171-208], is the Bayesian chapter and carries BRR, BayesA, BayesB,
BayesC and the Bayesian LASSO; BayesR is not among them and the string
"BayesR" does not occur anywhere in the book.

The method is therefore taken from the originating primary source, Erbe,
M., Hayes, B. J., Matukumalli, L. K., Goswami, S., Bowman, P. J.,
Reich, C. M., Mason, B. A. and Goddard, M. E. (2012), Improving accuracy
of genomic predictions within and between dairy cattle breeds with
imputed high-density single nucleotide polymorphism panels, *Journal of
Dairy Science* 95(7), 4114-4129, doi:10.3168/jds.2011-5019, which is the
paper that first states the four-class mixture.  Moser, G., Lee, S. H.,
Hayes, B. J., Goddard, M. E., Wray, N. R. and Visscher, P. M. (2015),
Simultaneous discovery, estimation and prediction analysis of complex
traits using a Bayesian mixture model, *PLoS Genetics* 11(4), e1004969,
doi:10.1371/journal.pgen.1004969, restates it and names it BayesR.

CITATION LIMIT, stated rather than papered over.  The Erbe et al. paper
is paywalled and its own text was not read.  The four-class
specification used here -- the mixture

    beta_j ~ pi_1 N(0, 0)          + pi_2 N(0, 1e-4 * sigma_g^2)
           + pi_3 N(0, 1e-3 * sigma_g^2) + pi_4 N(0, 1e-2 * sigma_g^2),
    (pi_1, ..., pi_4) ~ Dirichlet(delta), delta = (1, 1, 1, 1),

with the first class a point mass at zero -- is taken from Moser et
al.'s verbatim restatement of it, not from Erbe et al. directly.  The
multipliers are NOT assumed: they are the 0, 0.0001, 0.001 and 0.01 that
Moser et al. print.

DETERMINISM.  Nothing is sampled.  The Gibbs sampler is replaced by its
EM fixed point, which is exact and identical in both arms.  For marker j
with current residual r and column x, the class-conditional marginal
likelihood is the standard one for this mixture,

    L_k proportional to pi_k * sqrt(s_e2 / (x'x s_k2 + s_e2))
        * exp( (x'r)^2 s_k2 / (2 s_e2 (x'x s_k2 + s_e2)) ),

which at s_1^2 = 0 collapses to L_1 proportional to pi_1, the point mass;
gamma_jk is L_k normalised; the coefficient is its posterior mean
beta_j = sum_k gamma_jk * (x'r) s_k2 / (x'x s_k2 + s_e2); and the weights
are the Dirichlet posterior mean pi_k = (sum_j gamma_jk + delta_k) /
(p + sum delta).
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["bayes_r_prior"]

_CLASSES = (0.0, 1e-4, 1e-3, 1e-2)


def bayes_r_prior(y, X, pi=None, sigma_classes=None, max_iter=500, tol=1e-13,
                  delta=None):
    """BayesR fitted at the EM fixed point of its own Gibbs sampler.

    Parameters
    ----------
    y : array-like
        Length-n phenotypes.
    X : array-like
        n-by-p marker matrix.
    pi : array-like, optional
        Starting mixture weights; equal weights when absent.
    sigma_classes : array-like, optional
        The class variance multipliers of sigma_g^2.  Default
        (0, 1e-4, 1e-3, 1e-2), the four Moser et al. print.  The first
        entry must be exactly 0: it is the point mass that makes this
        BayesR rather than a plain scale mixture.
    max_iter, tol : int, float
        Fixed-point controls.
    delta : array-like, optional
        The Dirichlet parameter; (1, ..., 1) when absent.

    Returns
    -------
    estimate    : sigma_g^2, the total genetic variance
    beta_samples: the p marker effects at the fixed point
    class_probs : the n-by-K matrix of posterior class probabilities
    pi          : the fitted mixture weights
    sigma_g2    : the same as estimate
    sigma_e2    : the residual variance
    n_nonzero   : the number of markers whose modal class is not the spike
    """
    yy = core.vec(y)
    n = len(yy)
    if n < 2:
        raise ValueError("bayes_r_prior: need at least two observations")
    XX = core.mat(X)
    if len(XX) != n:
        raise ValueError("bayes_r_prior: X has a different number of rows than y")
    p = len(XX[0])
    if p < 1:
        raise ValueError("bayes_r_prior: X has no columns")
    for r in XX:
        if len(r) != p:
            raise ValueError("bayes_r_prior: X rows have unequal lengths")
    sc = list(_CLASSES) if sigma_classes is None else [float(v) for v in core.vec(sigma_classes)]
    K = len(sc)
    if K < 2:
        raise ValueError("bayes_r_prior: need at least two variance classes")
    if sc[0] != 0.0:
        raise ValueError("bayes_r_prior: the first variance class must be exactly 0, "
                         "the point mass that defines BayesR")
    for v in sc:
        if v < 0.0:
            raise ValueError("bayes_r_prior: variance class multipliers must be non-negative")
    if pi is None:
        pv = [1.0 / K] * K
    else:
        pv = [float(v) for v in core.vec(pi)]
        if len(pv) != K:
            raise ValueError("bayes_r_prior: pi must have one entry per variance class")
        if min(pv) < 0.0:
            raise ValueError("bayes_r_prior: mixture weights must be non-negative")
        t = sum(pv)
        if t <= 0.0:
            raise ValueError("bayes_r_prior: mixture weights must sum to something positive")
        pv = [v / t for v in pv]
    dl = [1.0] * K if delta is None else [float(v) for v in core.vec(delta)]
    if len(dl) != K:
        raise ValueError("bayes_r_prior: delta must have one entry per variance class")
    if min(dl) <= 0.0:
        raise ValueError("bayes_r_prior: the Dirichlet parameter must be positive")
    mu = sum(yy) / n
    beta = [0.0] * p
    xtx = [sum(XX[i][j] * XX[i][j] for i in range(n)) for j in range(p)]
    vy = sum((v - mu) ** 2 for v in yy) / (n - 1)
    sg2 = max(vy / 2.0, 1e-12)
    se2 = max(vy / 2.0, 1e-12)
    gam = [[0.0] * K for _ in range(p)]
    res = [yy[i] - mu for i in range(n)]
    for _ in range(int(max_iter)):
        prev = list(beta)
        for j in range(p):
            if xtx[j] <= 0.0:
                beta[j] = 0.0
                gam[j] = [1.0 if k == 0 else 0.0 for k in range(K)]
                continue
            # add marker j back into the residual
            bj = beta[j]
            if bj != 0.0:
                for i in range(n):
                    res[i] += XX[i][j] * bj
            xr = sum(XX[i][j] * res[i] for i in range(n))
            lg = []
            for k in range(K):
                s2 = sc[k] * sg2
                den = xtx[j] * s2 + se2
                lg.append(math.log(pv[k]) if pv[k] > 0.0 else -1e300)
                lg[k] += 0.5 * math.log(se2 / den) + xr * xr * s2 / (2.0 * se2 * den)
            mx = max(lg)
            w = [math.exp(v - mx) for v in lg]
            tw = sum(w)
            gam[j] = [v / tw for v in w]
            nb = 0.0
            for k in range(K):
                s2 = sc[k] * sg2
                nb += gam[j][k] * xr * s2 / (xtx[j] * s2 + se2)
            beta[j] = nb
            if nb != 0.0:
                for i in range(n):
                    res[i] -= XX[i][j] * nb
        # Dirichlet posterior mean of the weights
        tot = sum(dl)
        for k in range(K):
            pv[k] = (sum(gam[j][k] for j in range(p)) + dl[k]) / (p + tot)
        # variance components
        ss = 0.0
        wsum = 0.0
        for j in range(p):
            for k in range(1, K):
                if sc[k] > 0.0:
                    ss += gam[j][k] * beta[j] * beta[j] / sc[k]
                    wsum += gam[j][k]
        if wsum > 0.0:
            sg2 = max(ss / wsum, 1e-12)
        se2 = max(sum(v * v for v in res) / n, 1e-12)
        d = 0.0
        for j in range(p):
            d = max(d, abs(beta[j] - prev[j]))
        if d < tol:
            break
    nz = 0
    for j in range(p):
        mk = 0
        for k in range(K):
            if gam[j][k] > gam[j][mk]:
                mk = k
        if mk != 0:
            nz += 1
    return RichResult(
        title="BayesR",
        summary_lines=[("observations", n), ("markers", p), ("classes", K)],
        payload={
            "estimate": sg2,
            "beta_samples": beta,
            "class_probs": gam,
            "pi": pv,
            "sigma_g2": sg2,
            "sigma_e2": se2,
            "sigma_classes": sc,
            "mu": mu,
            "n_nonzero": nz,
            "n": n,
            "method": "BayesR four-class mixture of Erbe et al. (2012), at the EM fixed "
                      "point of its Gibbs sampler; not in the book",
        },
    )


def cheatsheet():
    return "baysr: BayesR: mixture of normals prior with different variance classes"


# compact alias per ledger/NAMING.md
bayesrprior = bayes_r_prior
