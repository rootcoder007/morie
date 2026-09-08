# morie.fn -- function file (rootcoder007/morie)
"""Optimal-transport domain adaptation by barycentric mapping.

Source FETCHED and read: Courty, N., Flamary, R., Tuia, D. &
Rakotomamonjy, A. (2017), "Optimal transport for domain adaptation",
*IEEE TPAMI* 39(9):1853-1865, arXiv:1507.00504, retrieved as
https://ar5iv.labs.arxiv.org/html/1507.00504 .

Equation (9) of that paper is the entropy-regularised transport problem
over the transport polytope with uniform marginals, which the paper
says "can be solved using the efficient Sinkhorn-Knopp algorithm".
Equation (14) is the barycentric mapping used to move the source
samples, quoted verbatim from the paper:

    Xs_hat = T_{gamma_0}(Xs) = diag(gamma_0 1_{nt})^-1 gamma_0 Xt

and equation (15) notes that for uniform marginals this reduces to
Xs_hat = ns gamma_0 Xt.  The cost here is the squared Euclidean
distance, the case for which the paper says the barycentre
"corresponds to a weighted average and the sample is mapped into the
convex hull of the target samples" (below eq. 13).

The Sinkhorn scaling runs a FIXED number of iterations rather than
stopping on a tolerance, so that the Python and R arms perform the
identical arithmetic.  ``epsilon`` is the regularisation weight, i.e.
1/lambda in the paper's notation.
"""

import math

from ._richresult import RichResult

__all__ = ["ot_domain_adaptation"]


def ot_domain_adaptation(Xs, Xt, epsilon, n_iter=1000):
    """Adapt source samples to a target domain through an OT plan.

    Parameters
    ----------
    Xs : sequence of sequences, shape (ns, d)
        Source samples.
    Xt : sequence of sequences, shape (nt, d)
        Target samples, same dimension d.
    epsilon : float
        Entropic regularisation weight (1/lambda in the paper).  Small
        values sharpen the plan but underflow exp(-C/epsilon).
    n_iter : int
        Fixed number of Sinkhorn-Knopp scaling sweeps.

    Returns
    -------
    RichResult
        Keys ``Xs_adapted``, ``gamma``, ``cost``, ``transport_cost``,
        ``row_error``, ``col_error``, ``ns``, ``nt``, ``d``,
        ``epsilon``, ``n_iter``, ``method``.
    """
    Xs = [[float(v) for v in r] for r in Xs]
    Xt = [[float(v) for v in r] for r in Xt]
    ns = len(Xs)
    nt = len(Xt)
    if ns == 0 or nt == 0:
        raise ValueError("both sample sets must be non-empty")
    d = len(Xs[0])
    for r in Xs + Xt:
        if len(r) != d:
            raise ValueError("Xs and Xt must share the same dimension")
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    C = [[sum((Xs[i][k] - Xt[j][k]) ** 2 for k in range(d))
          for j in range(nt)] for i in range(ns)]
    K = [[math.exp(-C[i][j] / epsilon) for j in range(nt)] for i in range(ns)]
    a = 1.0 / ns
    b = 1.0 / nt
    u = [1.0] * ns
    v = [1.0] * nt
    for _n in range(int(n_iter)):
        for i in range(ns):
            s = sum(K[i][j] * v[j] for j in range(nt))
            if s <= 0.0:
                raise ValueError("Sinkhorn kernel underflowed; raise epsilon")
            u[i] = a / s
        for j in range(nt):
            s = sum(K[i][j] * u[i] for i in range(ns))
            if s <= 0.0:
                raise ValueError("Sinkhorn kernel underflowed; raise epsilon")
            v[j] = b / s
    gamma = [[u[i] * K[i][j] * v[j] for j in range(nt)] for i in range(ns)]

    rows = [sum(gamma[i]) for i in range(ns)]
    cols = [sum(gamma[i][j] for i in range(ns)) for j in range(nt)]
    # equation (14): diag(gamma 1_nt)^-1 gamma Xt
    adapted = []
    for i in range(ns):
        if rows[i] <= 0.0:
            raise ValueError("a source point received no transported mass")
        adapted.append([sum(gamma[i][j] * Xt[j][k] for j in range(nt))
                        / rows[i] for k in range(d)])
    return RichResult(
        payload={
            "Xs_adapted": adapted,
            "gamma": gamma,
            "cost": C,
            "transport_cost": sum(gamma[i][j] * C[i][j]
                                  for i in range(ns) for j in range(nt)),
            "row_error": max(abs(rows[i] - a) for i in range(ns)),
            "col_error": max(abs(cols[j] - b) for j in range(nt)),
            "ns": ns,
            "nt": nt,
            "d": d,
            "epsilon": epsilon,
            "n_iter": int(n_iter),
            "method": "entropic OT domain adaptation, Courty et al (2017) "
                      "eq. (9) and (14)",
        }
    )


def cheatsheet():
    return "otdom: Joint distribution OT-based domain adaptation transform"
