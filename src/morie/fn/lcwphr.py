# morie.fn -- function file (rootcoder007/morie)
r"""Causal effects inside latent classes.

Latent class analysis finds subgroups that were never measured directly
-- patterns of co-occurring symptoms, of offending, of consumption --
from a set of categorical indicators. The temptation is then to compare
a treated and an untreated group *within* each recovered class and call
the difference an effect. It is not: the classes were not randomised and
neither was the treatment, so the comparison inherits every confounder
that made treatment more likely for some kinds of people than others.

Lanza, Coffman and Xu's answer is to keep the two pieces separate and
weight between them. The measurement model is fitted first:

.. math:: P(\mathbf y_i) = \sum_{j=1}^K \pi_j
          \prod_{q=1}^{Q} \rho_{qj}^{\,y_{iq}}(1-\rho_{qj})^{1-y_{iq}},

giving each subject a posterior class membership. A propensity model for
treatment given the same indicators supplies stabilised inverse
probability weights, and the class-specific effect is the difference of
two weighted means of the outcome, with each subject counted in a class
in proportion to its posterior membership rather than assigned to the
class it most resembles. The naive unweighted contrast is computed at
the same time, because the whole claim is that the two differ when
treatment was confounded and coincide when it was not.

The local independence assumption -- indicators independent given the
class -- is the model, not an approximation to it, and the classes are
returned in a canonical order (by prevalence) because a latent class
model is identified only up to relabelling.

References
----------
Lanza, S. T., Coffman, D. L. and Xu, S. (2013) "Causal inference in
latent class analysis", *Structural Equation Modeling* **20**(3),
361-383, doi:10.1080/10705511.2013.797816.

Lanza, S. T., Moore, J. E. and Butera, N. M. (2013) "Drawing causal
inferences using prevention science methodology", *Prevention Science*
**14**(5), 457-466, doi:10.1007/s11121-012-0334-x.

Goodman, L. A. (1974) "Exploratory latent structure analysis using both
identifiable and unidentifiable models", *Biometrika* **61**(2),
215-231, doi:10.1093/biomet/61.2.215.

Collins, L. M. and Lanza, S. T. (2010) *Latent Class and Latent
Transition Analysis*, Wiley, Ch. 2-4,
doi:10.1002/9780470567333.

Robins, J. M., Hernan, M. A. and Brumback, B. (2000) "Marginal
structural models and causal inference in epidemiology", *Epidemiology*
**11**(5), 550-560, doi:10.1097/00001648-200009000-00011. The
stabilised weights.

Cole, S. R. and Hernan, M. A. (2008) "Constructing inverse probability
weights for marginal structural models", *American Journal of
Epidemiology* **168**(6), 656-664, doi:10.1093/aje/kwn164.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["latent_class_weighted"]

_EPS = 1e-12


def _logit_irls(X, y, max_iter=100, tol=1e-11, ridge_rel=1e-8):
    """Logistic regression by IRLS; the ridge is scaled to the design."""
    n = len(y)
    p = len(X[0])
    beta = [0.0] * p
    for _ in range(max_iter):
        eta = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
        mu = [1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, e))))
              for e in eta]
        w = [max(mu[i] * (1.0 - mu[i]), 1e-10) for i in range(n)]
        z = [eta[i] + (y[i] - mu[i]) / w[i] for i in range(n)]
        A = [[sum(w[i] * X[i][a] * X[i][b] for i in range(n))
              for b in range(p)] for a in range(p)]
        scale = sum(A[a][a] for a in range(p)) / p
        for a in range(p):
            A[a][a] += ridge_rel * max(scale, _EPS)
        rhs = [sum(w[i] * X[i][a] * z[i] for i in range(n)) for a in range(p)]
        new = k.cholsolve(A, rhs)
        shift = max(abs(new[a] - beta[a]) for a in range(p))
        beta = new
        if shift < tol:
            break
    eta = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    return beta, [1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, e))))
                  for e in eta]


def latent_class_weighted(y, A, H, K, trim=0.0, stabilize=True,
                          max_iter=500, tol=1e-11):
    r"""Latent class analysis with inverse-probability-weighted effects.

    Parameters
    ----------
    y : array-like, length ``n``
        Distal outcome.
    A : array-like, length ``n``
        Binary treatment.
    H : array-like, shape ``(n, Q)``
        Binary manifest indicators defining the latent classes; also the
        covariates of the propensity model.
    K : int
        Number of latent classes.
    trim : float
        Symmetric trimming of the propensity score. ``0`` leaves the
        weights untouched; a positive value bounds them, which trades
        bias for the variance a near-zero propensity otherwise produces.

    Returns
    -------
    RichResult
        ``class_prevalence``, ``item_probabilities``, ``posterior``,
        the weighted ``class_ate`` and ``ate``, and the unweighted
        ``naive_class_ate``/``naive_ate`` they must be compared with.
    """
    yv = [float(v) for v in k.vec(y)]
    av = [float(v) for v in k.vec(A)]
    Hm = [[float(v) for v in row] for row in k.mat(H)]
    n = len(yv)
    if n == 0:
        raise ValueError("lcwphr: no observations")
    if len(av) != n or len(Hm) != n:
        raise ValueError("lcwphr: y, A and H must agree in length "
                         "(%d, %d, %d)" % (n, len(av), len(Hm)))
    Q = len(Hm[0])
    if any(len(r) != Q for r in Hm):
        raise ValueError("lcwphr: every row of H must have %d indicators" % Q)
    for i in range(n):
        for q in range(Q):
            if Hm[i][q] not in (0.0, 1.0):
                raise ValueError("lcwphr: the manifest indicators must be "
                                 "binary; H[%d][%d] = %r" % (i, q, Hm[i][q]))
    if any(v not in (0.0, 1.0) for v in av):
        raise ValueError("lcwphr: the treatment must be binary")
    if not any(v > 0.5 for v in av) or not any(v < 0.5 for v in av):
        raise ValueError("lcwphr: both treatment arms must be occupied -- "
                         "no contrast is identified from one arm")
    K = int(K)
    if K < 1:
        raise ValueError("lcwphr: K must be at least 1")
    tr = float(trim)
    if not 0.0 <= tr < 0.5:
        raise ValueError("lcwphr: trim must be in [0, 0.5)")

    # deterministic initialisation: rank subjects by how many indicators
    # they endorse and cut into K groups. No random start to disagree over.
    tot = [sum(Hm[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: (tot[i], i))
    lab0 = [0] * n
    for rank, i in enumerate(order):
        lab0[i] = min(rank * K // n, K - 1)
    pi = [0.0] * K
    rho = [[0.0] * Q for _ in range(K)]
    for j in range(K):
        idx = [i for i in range(n) if lab0[i] == j] or [order[min(j, n - 1)]]
        pi[j] = len(idx) / float(n)
        for q in range(Q):
            # shrunk towards 1/2 so no probability starts at a boundary,
            # where the EM update has nowhere to move
            s = sum(Hm[i][q] for i in idx)
            rho[j][q] = (s + 0.5) / (len(idx) + 1.0)

    post = [[0.0] * K for _ in range(n)]
    ll = -float("inf")
    path = []
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        ll_new = 0.0
        for i in range(n):
            lp = []
            for j in range(K):
                s = math.log(max(pi[j], 1e-300))
                for q in range(Q):
                    r = min(max(rho[j][q], 1e-12), 1.0 - 1e-12)
                    s += (math.log(r) if Hm[i][q] > 0.5
                          else math.log(1.0 - r))
                lp.append(s)
            mx = max(lp)
            tot_i = sum(math.exp(v - mx) for v in lp)
            ll_new += mx + math.log(tot_i)
            for j in range(K):
                post[i][j] = math.exp(lp[j] - mx) / tot_i
        path.append(ll_new)
        if it > 1 and abs(ll_new - ll) <= tol * (abs(ll) + 1.0):
            ll = ll_new
            converged = True
            break
        ll = ll_new
        for j in range(K):
            nk = sum(post[i][j] for i in range(n))
            pi[j] = nk / n
            nk = max(nk, 1e-300)
            for q in range(Q):
                rho[j][q] = sum(post[i][j] * Hm[i][q]
                                for i in range(n)) / nk

    # canonical order: prevalence descending, ties by the first indicator
    ordk = sorted(range(K), key=lambda j: (-pi[j], -rho[j][0], j))
    pi = [pi[j] for j in ordk]
    rho = [rho[j] for j in ordk]
    post = [[post[i][j] for j in ordk] for i in range(n)]
    labels = [max(range(K), key=lambda j: (post[i][j], -j)) for i in range(n)]

    # ---- propensity for treatment given the same indicators
    Xp = [[1.0] + Hm[i] for i in range(n)]
    pbeta, ps = _logit_irls(Xp, av)
    if tr > 0.0:
        ps = [min(max(v, tr), 1.0 - tr) for v in ps]
    else:
        ps = [min(max(v, 1e-8), 1.0 - 1e-8) for v in ps]
    marg = sum(av) / n
    w = []
    for i in range(n):
        d = ps[i] if av[i] > 0.5 else 1.0 - ps[i]
        nmr = (marg if av[i] > 0.5 else 1.0 - marg) if stabilize else 1.0
        w.append(nmr / d)

    def contrast(weights):
        num1 = sum(weights[i] * av[i] * yv[i] for i in range(n))
        den1 = sum(weights[i] * av[i] for i in range(n))
        num0 = sum(weights[i] * (1.0 - av[i]) * yv[i] for i in range(n))
        den0 = sum(weights[i] * (1.0 - av[i]) for i in range(n))
        if den1 <= _EPS or den0 <= _EPS:
            return float("nan"), float("nan"), float("nan")
        return num1 / den1 - num0 / den0, num1 / den1, num0 / den0

    class_ate, class_m1, class_m0 = [], [], []
    naive_ate, naive_m1, naive_m0 = [], [], []
    for j in range(K):
        gw = [post[i][j] * w[i] for i in range(n)]
        a_, m1, m0 = contrast(gw)
        class_ate.append(a_)
        class_m1.append(m1)
        class_m0.append(m0)
        gu = [post[i][j] for i in range(n)]
        a_, m1, m0 = contrast(gu)
        naive_ate.append(a_)
        naive_m1.append(m1)
        naive_m0.append(m0)

    ate = sum(pi[j] * class_ate[j] for j in range(K))
    naive = sum(pi[j] * naive_ate[j] for j in range(K))
    marginal_ate, _m1, _m0 = contrast(w)
    unweighted_ate, _u1, _u0 = contrast([1.0] * n)

    nfree = K - 1 + K * Q
    bic = -2.0 * ll + nfree * math.log(n)
    ess = sum(w) ** 2 / max(sum(v * v for v in w), 1e-300)
    ent = -sum(post[i][j] * math.log(max(post[i][j], 1e-300))
               for i in range(n) for j in range(K))

    return RichResult(payload={
        "estimate": class_ate, "class_ate": class_ate,
        "class_mean_treated": class_m1, "class_mean_control": class_m0,
        "naive_class_ate": naive_ate,
        "naive_class_mean_treated": naive_m1,
        "naive_class_mean_control": naive_m0,
        "ate": ate, "naive_ate": naive,
        "marginal_ate": marginal_ate, "unweighted_ate": unweighted_ate,
        "class_prevalence": pi, "item_probabilities": rho,
        "posterior": post, "labels": labels,
        "propensity": ps, "propensity_coefficients": pbeta,
        "weights": w, "effective_sample_size": ess,
        "weight_max": max(w), "weight_mean": sum(w) / n,
        "loglik": ll, "loglik_path": path, "bic": bic, "entropy": ent,
        "n_parameters": nfree, "iterations": it, "converged": converged,
        "K": K, "n": n, "Q": Q, "stabilized": bool(stabilize), "trim": tr,
        "method": "latent class analysis by EM with inverse-probability-"
                  "weighted class-specific treatment effects, subjects "
                  "counted in proportion to posterior membership "
                  "(Lanza, Coffman & Xu 2013; Robins et al. 2000)",
        "note": "class_ate and naive_class_ate coincide when treatment was "
                "unrelated to the indicators and separate when it was not "
                "-- the gap is what the weighting is for; classes are "
                "returned in prevalence order because the model is "
                "identified only up to relabelling",
    })


def cheatsheet():
    return ("lcwphr: latent_class_weighted(y, A, H, K) -> latent classes "
            "plus IPW class-specific treatment effects (Lanza, Coffman & "
            "Xu 2013, Structural Equation Modeling 20:361-383)")
