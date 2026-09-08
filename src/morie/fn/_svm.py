# morie.fn -- shared helpers (rootcoder007/morie)
"""SMO solver shared by the support-vector modules.

Platt's sequential minimal optimisation for the dual soft-margin problem

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
    raise ValueError(f'unknown kernel {kernel!r}; expected linear, poly, rbf or sigmoid')


def smo(K, y, C=1.0, tol=1e-3, max_passes=50, max_iter=10000, seed=0):
    """Simplified SMO. Returns ``(alpha, b, n_iter, converged)``.

    ``K`` is the training Gram matrix and ``y`` is in {-1, +1}.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    alpha = np.zeros(n)
    b = 0.0
    rng = np.random.default_rng(seed)
    passes = it = 0
    while passes < max_passes and it < max_iter:
        changed = 0
        for i in range(n):
            it += 1
            Ei = float(K[i] @ (alpha * y)) + b - y[i]
            if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
                j = int(rng.integers(n - 1))
                j = j + 1 if j >= i else j
                Ej = float(K[j] @ (alpha * y)) + b - y[j]
                ai_old, aj_old = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L, Hi = max(0.0, aj_old - ai_old), min(C, C + aj_old - ai_old)
                else:
                    L, Hi = max(0.0, ai_old + aj_old - C), min(C, ai_old + aj_old)
                if L >= Hi:
                    continue
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                alpha[j] = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, Hi)
                if abs(alpha[j] - aj_old) < 1e-12:
                    alpha[j] = aj_old
                    continue
                alpha[i] = ai_old + y[i] * y[j] * (aj_old - alpha[j])
                b1 = b - Ei - y[i] * (alpha[i] - ai_old) * K[i, i] \
                    - y[j] * (alpha[j] - aj_old) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - ai_old) * K[i, j] \
                    - y[j] * (alpha[j] - aj_old) * K[j, j]
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                changed += 1
        passes = passes + 1 if changed == 0 else 0
    return alpha, float(b), int(it), bool(passes >= max_passes)
