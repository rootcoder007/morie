"""Support vector data description (Tax & Duin 2004)."""

import math

from ._richresult import RichResult

__all__ = ["deepSVDD", "svdd", "support_vector_data_description"]


def _kernel(a, b, kern, gamma):
    if kern == "linear":
        return sum(x * y for x, y in zip(a, b))
    d2 = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.exp(-gamma * d2)


def svdd(X, C=1.0, kernel="linear", gamma=1.0, tol=1e-10,
         max_sweeps=500):
    """
    Support vector data description: minimal enclosing hypersphere.

    Tax & Duin (2004): minimize the sphere volume R^2 + C sum xi_i
    subject to |x_i - a|^2 <= R^2 + xi_i.  The dual (their Eq. 10)
    maximizes L = sum_i alpha_i K(x_i, x_i)
    - sum_ij alpha_i alpha_j K(x_i, x_j) subject to
    sum alpha_i = 1, 0 <= alpha_i <= C (Eqs. 6, 9).  The centre is
    a = sum_i alpha_i x_i (Eq. 7); R^2 is the distance to any
    boundary support vector (0 < alpha < C); a test point z is
    accepted when Eq. 14 holds.  The KKT conditions (Eqs. 11-13)
    characterize the solution: interior points have alpha = 0,
    boundary SVs 0 < alpha < C, outliers alpha = C.  Solved by
    deterministic pairwise coordinate ascent on the dual (each pair
    update is the exact 1-D quadratic maximizer clipped to the box,
    preserving sum alpha = 1).

    Sources
    -------
    Tax, D. M. J. & Duin, R. P. W. (2004). Support vector data
    description. *Machine Learning*, 54(1), 45-66, Eqs. 3-14
    (local copy fetched-wave3/Support Vector Data Description.pdf).

    Parameters
    ----------
    X : matrix (n x d)
        Training objects.
    C : float
        Box constraint (>= 1/n; C >= 1 disables outliers).
    kernel : str
        "linear" or "rbf".
    gamma : float
        RBF width parameter.
    tol, max_sweeps : convergence controls.

    Returns
    -------
    RichResult
        Keys: alpha, center (linear kernel), radius2, support
        (indices with alpha > tol), outliers (alpha ~ C),
        kkt_violation (max over Eqs. 11-13 checks).
    """
    Xv = [[float(v) for v in row] for row in X]
    n = len(Xv)
    if n < 2:
        raise ValueError("need at least two objects")
    C = float(C)
    if C < 1.0 / n:
        raise ValueError("need C >= 1/n for a feasible dual")
    kern = str(kernel).lower()
    if kern not in ("linear", "rbf"):
        raise ValueError("kernel must be 'linear' or 'rbf'")
    K = [[_kernel(Xv[i], Xv[j], kern, gamma) for j in range(n)]
         for i in range(n)]
    alpha = [1.0 / n] * n

    def _grad(i):
        # dL/dalpha_i = K_ii - 2 sum_j alpha_j K_ij
        return K[i][i] - 2.0 * sum(alpha[j] * K[i][j] for j in range(n))

    for _sweep in range(int(max_sweeps)):
        moved = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s = alpha[i] + alpha[j]
                lo = max(0.0, s - C)
                hi = min(C, s)
                if hi - lo < 1e-15:
                    continue
                # L(alpha_i) with alpha_j = s - alpha_i is quadratic;
                # stationary point:
                denom = 2.0 * (K[i][i] - 2.0 * K[i][j] + K[j][j])
                gi = _grad(i)
                gj = _grad(j)
                if denom <= 1e-300:
                    new = hi if gi - gj > 0 else lo
                else:
                    new = alpha[i] + (gi - gj) / denom
                new = min(max(new, lo), hi)
                delta = new - alpha[i]
                if abs(delta) > 1e-16:
                    alpha[i] = new
                    alpha[j] = s - new
                    moved = max(moved, abs(delta))
        if moved < tol:
            break
    sup = [i for i in range(n) if alpha[i] > 1e-8]
    boundary = [i for i in sup if alpha[i] < C - 1e-8]
    out = [i for i in range(n) if alpha[i] >= C - 1e-8]

    def _dist2(i):
        # |x_i - a|^2 in kernel space (Eq. 14 with z = x_i)
        return (K[i][i]
                - 2.0 * sum(alpha[j] * K[i][j] for j in range(n))
                + sum(alpha[a_] * alpha[b_] * K[a_][b_]
                      for a_ in sup for b_ in sup))

    if boundary:
        r2s = [_dist2(i) for i in boundary]
        radius2 = sum(r2s) / len(r2s)
        r2_spread = max(r2s) - min(r2s)
    else:
        radius2 = max(_dist2(i) for i in range(n))
        r2_spread = 0.0
    # KKT checks (Eqs. 11-13)
    viol = r2_spread
    for i in range(n):
        d2 = _dist2(i)
        if alpha[i] < 1e-8 and d2 > radius2 + 1e-6:
            viol = max(viol, d2 - radius2)
        if alpha[i] >= C - 1e-8 and C < 1.0 and d2 < radius2 - 1e-6:
            viol = max(viol, radius2 - d2)
    center = None
    if kern == "linear":
        d = len(Xv[0])
        center = [sum(alpha[i] * Xv[i][k] for i in range(n))
                  for k in range(d)]
    return RichResult(payload={
        "alpha": alpha,
        "center": center,
        "radius2": radius2,
        "support": sup,
        "outliers": out,
        "kkt_violation": viol,
        "kernel": kern,
        "C": C,
        "method": "SVDD (Tax & Duin 2004, Eqs. 6-14)",
    })


# stub-era and worklist names
deepSVDD = svdd
support_vector_data_description = svdd


def cheatsheet():
    return "svdd: max sum a K_ii - aa'K, sum a=1, 0<=a<=C; a = center weights"

# public names resolved by fn/_lazy_map.json
deep_svdd = svdd
deepsvdd = svdd
