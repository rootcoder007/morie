# morie.fn -- shared helpers (rootcoder007/morie)
"""SMO solver shared by the support-vector modules.

Sequential minimal optimisation for the dual soft-margin problem

.. math::
    \\max_\\alpha \\sum_i \\alpha_i
        - \\tfrac12 \\sum_{i,j} \\alpha_i \\alpha_j y_i y_j K(x_i, x_j)
    \\quad\\text{s.t.}\\quad 0 \\le \\alpha_i \\le C, \\;\\sum_i \\alpha_i y_i = 0 .

Written once and shared so a sign error cannot exist in one kernel's copy
and not another's.
"""

from __future__ import annotations

from . import _array_core as np

__all__ = ["kernel_matrix", "smo"]


def kernel_matrix(X, Z=None, kernel="rbf", gamma=None, degree=3, coef0=1.0):
    """Gram matrix K(X, Z) for the supported kernels."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Z = X if Z is None else np.atleast_2d(np.asarray(Z, dtype=float))
    if X.shape[1] != Z.shape[1]:
        raise ValueError(f"X has {X.shape[1]} columns but Z has {Z.shape[1]}")
    if kernel == "linear":
        return X @ Z.T
    if kernel == "poly":
        return (X @ Z.T + coef0) ** degree
    if kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        d2 = (X**2).sum(1)[:, None] + (Z**2).sum(1)[None, :] - 2 * X @ Z.T
        return np.exp(-gamma * np.maximum(d2, 0.0))
    if kernel == "sigmoid":
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        return np.tanh(gamma * X @ Z.T + coef0)
    raise ValueError(f"unknown kernel {kernel!r}; expected linear, poly, rbf or sigmoid")


def smo(K, y, C=1.0, tol=1e-3, max_passes=50, max_iter=100000, seed=0, p=None):
    """SMO with the maximal-violating-pair working set. Returns ``(alpha, b, n_iter, converged)``.

    ``K`` is the training Gram matrix and ``y`` is in {-1, +1}. With the
    gradient ``G = Q alpha - 1`` (``Q_ij = y_i y_j K_ij``) the pair is
    ``i = argmax_{I_up} -y_t G_t`` and ``j = argmin_{I_low} -y_t G_t``
    (Keerthi et al. 2001; the working set of libsvm, Fan, Chen & Lin 2005),
    updated analytically within the box; the iteration stops when the
    violation ``m - M`` drops below ``tol``, which certifies the dual optimum.
    The bias is the average of ``-y_t G_t`` over the free support vectors
    (or the midpoint of the bounds). Deterministic: ``max_passes`` and
    ``seed`` are accepted for backward compatibility and not used. ``p`` is
    the linear term of the dual ``1/2 a'Qa + p'a`` (default all -1, the
    classifier); support-vector regression passes its own.
    """
    y = np.asarray(y, dtype=float).ravel()
    Km = np.asarray(K, dtype=float)
    n = y.size
    Kl = [[float(Km[i, j]) for j in range(n)] for i in range(n)]
    yl = [float(v) for v in y]
    alpha = [0.0] * n
    G = [-1.0] * n if p is None else [float(v) for v in p]
    it, converged = 0, False
    while it < max_iter:
        it += 1
        i, gmax = -1, -float("inf")
        j, gmin = -1, float("inf")
        for t in range(n):
            v = -yl[t] * G[t]
            up = (yl[t] > 0 and alpha[t] < C) or (yl[t] < 0 and alpha[t] > 0)
            low = (yl[t] > 0 and alpha[t] > 0) or (yl[t] < 0 and alpha[t] < C)
            if up and v > gmax:
                i, gmax = t, v
            if low and v < gmin:
                j, gmin = t, v
        if i < 0 or j < 0 or gmax - gmin < tol:
            converged = True
            break
        a = Kl[i][i] + Kl[j][j] - 2 * Kl[i][j]
        if a <= 0:
            a = 1e-12
        t = (gmax - gmin) / a
        # alpha_i += y_i t, alpha_j -= y_j t, kept inside [0, C]
        lo, hi = -float("inf"), float("inf")
        if yl[i] > 0:
            lo, hi = max(lo, -alpha[i]), min(hi, C - alpha[i])
        else:
            lo, hi = max(lo, alpha[i] - C), min(hi, alpha[i])
        if yl[j] > 0:
            lo, hi = max(lo, alpha[j] - C), min(hi, alpha[j])
        else:
            lo, hi = max(lo, -alpha[j]), min(hi, C - alpha[j])
        t = min(max(t, lo), hi)
        di, dj = yl[i] * t, -yl[j] * t
        alpha[i] += di
        alpha[j] += dj
        alpha[i] = min(max(alpha[i], 0.0), C)
        alpha[j] = min(max(alpha[j], 0.0), C)
        for k in range(n):
            G[k] += yl[k] * (yl[i] * Kl[k][i] * di + yl[j] * Kl[k][j] * dj)
    free = [-yl[t] * G[t] for t in range(n) if 0 < alpha[t] < C]
    if free:
        b = sum(free) / len(free)
    else:
        ub = [-yl[t] * G[t] for t in range(n) if (yl[t] > 0 and alpha[t] < C) or (yl[t] < 0 and alpha[t] > 0)]
        lb = [-yl[t] * G[t] for t in range(n) if (yl[t] > 0 and alpha[t] > 0) or (yl[t] < 0 and alpha[t] < C)]
        b = (min(ub) + max(lb)) / 2 if ub and lb else 0.0
    return np.asarray(alpha), float(b), int(it), bool(converged)
