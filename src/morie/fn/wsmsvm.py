# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linear SVM (hard margin via dual coordinate ascent)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_svm"]


def wasserman_svm(X, y, C=1e6, max_iter=100000, tol=1e-12):
    """
    Linear support vector machine.

    Formula: min (1/2)|w|^2 s.t. y_i (w'x_i + b) >= 1, solved
    through the box-constrained dual
    max sum a_i - (1/2) sum a_i a_j y_i y_j x_i'x_j, 0 <= a_i <= C,
    by deterministic cyclic pairwise (SMO-style) updates that keep
    sum a_i y_i = 0 exact. The default C = 1e6 approximates the hard
    margin; b is recovered from free support vectors
    (0 < a_i < C). Margin, support-vector set and KKT residual ship
    in the payload so optimality is inspectable, not asserted.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Features.
    y : array-like
        Labels in {-1, +1}, both classes present.
    C : float
        Box constraint (large = hard margin).
    max_iter, tol
        Solver controls.

    Returns
    -------
    result : dict
        Keys: estimate (margin 2/|w|), w, b, support_vectors
        (0-based indices), alphas, kkt_violation, n, d, method.

    References
    ----------
    Wasserman (2004), Ch 22 (support vector machines); Platt (1998).

    Examples
    --------
    Two points at (-1,-1), (1,1): max-margin plane is x1 + x2 = 0
    with w = (0.5, 0.5), margin 2/|w| = 2 sqrt(2).

    >>> out = wasserman_svm([[-1.0, -1.0], [1.0, 1.0]], [-1, 1])
    >>> [round(v, 10) for v in out["w"]]
    [0.5, 0.5]
    >>> round(out["b"], 10)
    0.0
    >>> round(out["estimate"], 10) == round(2 * 2 ** 0.5, 10)
    True
    >>> out["support_vectors"]
    [0, 1]
    >>> wasserman_svm([[0.0], [1.0]], [1, 1])
    Traceback (most recent call last):
        ...
    ValueError: the SVM needs both classes present.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, d = X.shape
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} labels.")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("labels must lie in {-1, +1}.")
    if np.all(y == y[0]):
        raise ValueError("the SVM needs both classes present.")
    C = float(C)
    K = X @ X.T
    a = np.zeros(n)
    # Pairwise dual ascent on c = a*y: for a pair (i, j) move
    # c_i += t, c_j -= t (keeps sum a_i y_i = 0). The exact optimum is
    # t* = (y_i - y_j - u_i + u_j) / kappa with u = K c and
    # kappa = K_ii - 2 K_ij + K_jj, then t is clipped to the box.
    for _ in range(int(max_iter)):
        moved = 0.0
        u = K @ (a * y)
        for i in range(n):
            for j in range(i + 1, n):
                kappa = K[i, i] - 2.0 * K[i, j] + K[j, j]
                if kappa <= 1e-300:
                    continue
                tstar = (y[i] - y[j] - u[i] + u[j]) / kappa
                # box on a_i: 0 <= a_i + y_i t <= C
                b1, b2 = -a[i] * y[i], (C - a[i]) * y[i]
                lo, hi = min(b1, b2), max(b1, b2)
                # box on a_j: 0 <= a_j - y_j t <= C
                b3, b4 = a[j] * y[j], (a[j] - C) * y[j]
                lo, hi = max(lo, min(b3, b4)), min(hi, max(b3, b4))
                tt = min(max(tstar, lo), hi)
                if tt == 0.0:
                    continue
                a[i] += y[i] * tt
                a[j] -= y[j] * tt
                du = K[:, i] * tt - K[:, j] * tt
                u += du
                moved += abs(tt)
        if moved < tol:
            break
    w = X.T @ (a * y)
    free = np.where((a > 1e-8) & (a < C - 1e-8))[0]
    sv = np.where(a > 1e-8)[0]
    if free.size:
        b = float(np.mean(y[free] - X[free] @ w))
    elif sv.size:
        b = float(np.mean(y[sv] - X[sv] @ w))
    else:
        raise ValueError("the solver found no support vectors; data may be degenerate.")
    margins = y * (X @ w + b)
    kkt = float(max(0.0, 1.0 - float(np.min(margins[sv])))) if sv.size else float("nan")
    norm = float(np.linalg.norm(w))
    return RichResult(payload={
        "estimate": float(2.0 / norm) if norm > 0 else float("inf"),
        "w": [float(v) for v in w], "b": b,
        "support_vectors": [int(v) for v in sv],
        "alphas": [float(v) for v in a],
        "kkt_violation": kkt, "n": int(n), "d": int(d),
        "method": "linear SVM, cyclic pairwise dual ascent, C=1e6 ~ hard margin"})


def cheatsheet():
    return "wsmsvm: dual ascent over ALL pairs i<j; w = X'(a*y); b from free SVs"


# compact alias per ledger/NAMING.md
wassermansvm = wasserman_svm
