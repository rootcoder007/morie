# morie.fn -- function file (rootcoder007/morie)
"""Multiparty competition under logit probabilistic voting (Schofield's model)."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._mlfa import eigh_desc
from ._qpcore import solve, ssum
from .bfgsmin import bfgs_minimize


def _probs(X, Z, lam, beta):
    P = []
    for x in X:
        u = [lam[j] - beta * ssum((a - b) ** 2 for a, b in zip(x, z)) for j, z in enumerate(Z)]
        top = max(u)
        e = [math.exp(t - top) for t in u]
        s = ssum(e)
        P.append([t / s for t in e])
    return P


def _share_and_grad(X, Z, lam, beta, j):
    P = _probs(X, Z, lam, beta)
    n, d = len(X), len(Z[j])
    share = ssum(p[j] for p in P) / n
    g = [ssum(2 * beta * P[i][j] * (1 - P[i][j]) * (X[i][k] - Z[j][k]) for i in range(n)) / n for k in range(d)]
    return share, g


def logit_competition(
    voters, valence, beta, *, start=None, tol: float = 1e-10, max_iter: int = 500
) -> DescriptiveResult:
    """Local Nash equilibrium of vote-share-maximising parties under logit voting.

    Voter ``i`` at ``x_i`` picks party ``j`` at ``z_j`` with probability
    ``exp(lambda_j - beta ||x_i - z_j||^2) / sum_k exp(lambda_k - beta ||x_i - z_k||^2)``
    (valences ``lambda``); party ``j`` maximises its expected share
    ``V_j = mean_i P_ij``. The equilibrium is found by alternating best
    responses (BFGS on the analytic gradients) from ``start`` (default:
    the parties spread on a circle of radius ``sqrt(trace V) / 2`` about
    the electoral mean), polished by Newton steps on the
    joint first-order conditions. Schofield (2007): with the electoral
    covariance ``V`` (divisor n) and ``rho_j`` the share of party ``j``
    when all sit at the mean, the Hessian of ``V_j`` there is
    ``2 beta rho_j (1 - rho_j) C_j`` with ``C_j = 2 beta (1 - 2 rho_j) V - I``,
    so the mean is a local Nash equilibrium iff every ``C_j`` is negative
    definite; ``c = 2 beta (1 - 2 rho_min) trace(V)`` is his convergence
    coefficient.

    :param voters: Voter ideal points (n x d).
    :param valence: Party valences ``lambda_j``.
    :param beta: Spatial salience (> 0).
    :param start: Optional starting positions (p x d).
    :param tol: Stop when no party moves more than ``tol``.
    :param max_iter: Maximum best-response rounds.
    :return: DescriptiveResult; ``value`` is the positions; ``extra`` has
        ``shares``, ``gradients``, ``hessian_eigenvalues`` (of each party's
        share at the solution, numerically), ``is_local_nash``,
        ``characteristic_matrices`` ``C_j`` and ``mean_shares`` ``rho_j``
        at the mean, ``convergence_coefficient``, ``rounds``, ``converged``.

    References
    ----------
    Schofield, N. (2007). The mean voter theorem: necessary and sufficient
    conditions for convergent equilibrium. Review of Economic Studies 74,
    965-980.

    Lin, T.-M., Enelow, J. M. and Dorussen, H. (1999). Equilibrium in
    multicandidate probabilistic spatial voting. Public Choice 98, 59-82.

    Examples
    --------
    >>> X = [[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, -2.0]]
    >>> r = logit_competition(X, [0.0, 0.0, 0.0], 0.1)
    >>> [[round(t, 9) + 0.0 for t in z] for z in r.value], r.extra["is_local_nash"]
    ([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], True)
    """
    X = [[float(t) for t in x] for x in voters]
    lam = [float(t) for t in valence]
    beta = float(beta)
    n, d, p = len(X), len(X[0]), len(lam)
    mean = [ssum(x[k] for x in X) / n for k in range(d)]
    V = [[ssum((x[a] - mean[a]) * (x[b] - mean[b]) for x in X) / n for b in range(d)] for a in range(d)]
    top = max(lam)
    e = [math.exp(t - top) for t in lam]
    rho = [t / ssum(e) for t in e]
    C = [
        [[2 * beta * (1 - 2 * r) * V[a][b] - (1.0 if a == b else 0.0) for b in range(d)] for a in range(d)] for r in rho
    ]
    if start is not None:
        Z = [list(map(float, z)) for z in start]
    else:
        rad = 0.5 * math.sqrt(ssum(V[a][a] for a in range(d)))
        Z = []
        for j in range(p):
            th = 2 * math.pi * j / p
            off = [math.cos(th), math.sin(th)] + [0.0] * (d - 2) if d > 1 else [1.0 if j % 2 == 0 else -1.0]
            Z.append([m + rad * o for m, o in zip(mean, off)])
    rounds, converged = 0, False
    while rounds < max_iter:
        rounds += 1
        move = 0.0
        for j in range(p):

            def f(z, j=j):
                return -_share_and_grad(X, Z[:j] + [z] + Z[j + 1 :], lam, beta, j)[0]

            def g(z, j=j):
                return [-t for t in _share_and_grad(X, Z[:j] + [z] + Z[j + 1 :], lam, beta, j)[1]]

            b = bfgs_minimize(f, Z[j], grad=g, gtol=1e-12)["x"]
            move = max(move, max(abs(s - t) for s, t in zip(b, Z[j])))
            Z[j] = list(b)
        if move <= tol:
            converged = True
            break

    def foc(w):
        W = [w[j * d : (j + 1) * d] for j in range(p)]
        out = []
        for j in range(p):
            out += _share_and_grad(X, W, lam, beta, j)[1]
        return out

    w = [t for z in Z for t in z]
    for _ in range(20):
        F = foc(w)
        if max(abs(t) for t in F) <= 1e-14:
            break
        J = [[0.0] * (p * d) for _ in range(p * d)]
        for c in range(p * d):
            wp, wm = list(w), list(w)
            wp[c] += 1e-6
            wm[c] -= 1e-6
            Fp, Fm = foc(wp), foc(wm)
            for r in range(p * d):
                J[r][c] = (Fp[r] - Fm[r]) / 2e-6
        w = [a + s for a, s in zip(w, solve(J, [-t for t in F]))]
    Z = [w[j * d : (j + 1) * d] for j in range(p)]
    eig, shares, grads = [], [], []
    for j in range(p):
        s, gj = _share_and_grad(X, Z, lam, beta, j)
        shares.append(s)
        grads.append(gj)
        H = [[0.0] * d for _ in range(d)]
        for c in range(d):
            zp, zm = list(Z[j]), list(Z[j])
            zp[c] += 1e-5
            zm[c] -= 1e-5
            gp = _share_and_grad(X, Z[:j] + [zp] + Z[j + 1 :], lam, beta, j)[1]
            gm = _share_and_grad(X, Z[:j] + [zm] + Z[j + 1 :], lam, beta, j)[1]
            for r in range(d):
                H[r][c] = (gp[r] - gm[r]) / 2e-5
        H = [[(H[a][b] + H[b][a]) / 2 for b in range(d)] for a in range(d)]
        eig.append(eigh_desc(H)[0])
    return DescriptiveResult(
        name="logit_competition",
        value=Z,
        extra={
            "shares": shares,
            "gradients": grads,
            "hessian_eigenvalues": eig,
            "is_local_nash": all(max(ev) < 0 for ev in eig) and max(abs(t) for gj in grads for t in gj) < 1e-8,
            "characteristic_matrices": C,
            "mean_shares": rho,
            "convergence_coefficient": 2 * beta * (1 - 2 * min(rho)) * ssum(V[a][a] for a in range(d)),
            "rounds": rounds,
            "converged": converged,
        },
    )


logcmp = logit_competition


def cheatsheet() -> str:
    return "logit_competition(voters, valence, beta) -> Schofield multiparty logit local Nash equilibrium"
