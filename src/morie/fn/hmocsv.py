# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-class SVM: learn the boundary of the high-density region."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_one_class_svm"]


def _rbf(A, B, gamma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)


def geron_one_class_svm(X, nu=0.5, gamma=1.0, max_iter=2000, tol=1e-9):
    """
    One-class SVM: learn the boundary of the high-density region.

    Formula: find the smallest sphere / hyperplane separating the data from the origin

    The dual is minimise (1/2) a^T K a subject to 0 <= a_i <= 1/(nu*n)
    and sum a_i = 1, solved here by SMO: pairs of multipliers are moved
    against each other so the sum constraint holds exactly at every step.

    nu is the knob with an actual meaning -- it upper-bounds the fraction
    of training points left outside the boundary and lower-bounds the
    fraction that become support vectors. The box 1/(nu*n) is what
    enforces it: no single point can absorb more than that much weight,
    so at small nu the boundary has to stretch around outliers instead of
    excluding them.

    With an RBF kernel every point has unit norm in feature space, so the
    separating hyperplane and the smallest enclosing sphere are the same
    problem.

    Parameters
    ----------
    X : array-like, shape (n, d)
    nu : float, default 0.5
        In (0, 1]; ``nu * n`` must be at least 1.
    gamma : float, default 1.0
        RBF width (positive).
    max_iter : int, default 2000
    tol : float, default 1e-9
        KKT violation tolerance.

    Returns
    -------
    result : RichResult
        Keys: alpha, rho, decision, is_outlier, support_vectors,
        outlier_fraction, decision_function, estimate, n, method.

    Examples
    --------
    Five points around the origin and one far away: the far point has the
    lowest decision value and is excluded.

    >>> X = [[0.0], [0.05], [-0.05], [0.1], [-0.1], [5.0]]
    >>> r = geron_one_class_svm(X, nu=0.5, gamma=1.0)
    >>> int(np.argmin(r["decision"]))
    5
    >>> bool(r["is_outlier"][5]) and not bool(r["is_outlier"][0])
    True

    nu caps the training outlier fraction:

    >>> bool(r["outlier_fraction"] <= 0.5 + 1e-9)
    True

    The decision function extends to new points:

    >>> bool(r["decision_function"]([[0.0]])[0] > r["decision_function"]([[9.0]])[0])
    True

    References
    ----------
    Geron Ch 8
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"geron_one_class_svm: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_one_class_svm: X contains non-finite values")
    n = A.shape[0]
    v = float(nu)
    if not (0.0 < v <= 1.0):
        raise ValueError(f"geron_one_class_svm: nu must lie in (0, 1], got {nu!r}")
    C = 1.0 / (v * n)
    if C < 1.0 / n - 1e-12:
        raise ValueError(f"geron_one_class_svm: nu*n = {v * n} must be at least 1 for the constraints to be feasible")
    g = float(gamma)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"geron_one_class_svm: gamma must be positive and finite, got {gamma!r}")
    it_max = int(max_iter)
    if it_max < 1:
        raise ValueError(f"geron_one_class_svm: max_iter must be >= 1, got {max_iter!r}")

    K = _rbf(A, A, g)
    alpha = np.zeros(n)
    full = int(np.floor(1.0 / C))
    alpha[:full] = C
    if full < n:
        alpha[full] = 1.0 - C * full
    grad = K @ alpha

    n_iter = 0
    for n_iter in range(1, it_max + 1):
        up = np.where(alpha < C - 1e-12)[0]
        dn = np.where(alpha > 1e-12)[0]
        if up.size == 0 or dn.size == 0:
            break
        i = up[np.argmin(grad[up])]
        j = dn[np.argmax(grad[dn])]
        gap = grad[j] - grad[i]
        if gap <= tol:
            break
        denom = K[i, i] + K[j, j] - 2.0 * K[i, j]
        if denom <= 1e-15:
            step = min(C - alpha[i], alpha[j])
        else:
            step = min(gap / denom, C - alpha[i], alpha[j])
        if step <= 0:
            break
        alpha[i] += step
        alpha[j] -= step
        grad += step * (K[:, i] - K[:, j])

    # rho by the libsvm convention: the free support vectors pin it exactly;
    # with none free it is the midpoint of the bracketing gradients, which is
    # what keeps the nu-property (outlier fraction <= nu) when every
    # multiplier sits at a bound.
    free = np.where((alpha > 1e-9) & (alpha < C - 1e-9))[0]
    if free.size:
        rho = float(np.mean(grad[free]))
    else:
        at_box = grad[alpha >= C - 1e-9]
        at_zero = grad[alpha <= 1e-9]
        lb = float(np.max(at_box)) if at_box.size else -np.inf
        ub = float(np.min(at_zero)) if at_zero.size else np.inf
        if np.isfinite(lb) and np.isfinite(ub):
            rho = 0.5 * (lb + ub)
        elif np.isfinite(lb):
            rho = lb
        elif np.isfinite(ub):
            rho = ub
        else:
            rho = float(np.mean(grad))
    decision = grad - rho
    outlier = decision < 0

    def decision_function(Xnew, _A=A, _a=alpha, _g=g, _r=rho, _d=A.shape[1]):
        B = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if B.shape[1] != _d:
            raise ValueError(f"decision_function: expected {_d} features, got {B.shape[1]}")
        return _rbf(B, _A, _g) @ _a - _r

    return RichResult(
        title="One-class SVM",
        summary_lines=[("nu", v), ("Support vectors", int(np.sum(alpha > 1e-9))), ("Outlier fraction", float(np.mean(outlier)))],
        interpretation="nu bounds the outlier fraction above and the support-vector fraction below.",
        payload={
            "alpha": alpha,
            "rho": rho,
            "decision": decision,
            "is_outlier": outlier,
            "support_vectors": np.where(alpha > 1e-9)[0],
            "outlier_fraction": float(np.mean(outlier)),
            "decision_function": decision_function,
            "n_iter": int(n_iter),
            "C": C,
            "estimate": decision,
            "n": int(n),
            "method": "One-class SVM dual solved by SMO with an RBF kernel",
        },
    )


def cheatsheet():
    return "hmocsv: One-class SVM boundary of the high-density region"
