# morie.fn -- function file (rootcoder007/morie)
"""Kernel support vector machine -- ESL Sec 12.3."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._svm import kernel_matrix, smo

__all__ = ["esl_svm_kernel"]


def esl_svm_kernel(X, y, C=1.0, kernel="rbf", gamma=None, degree=3, coef0=1.0,
                   newdata=None, tol=1e-3, max_passes=50, seed=0):
    r"""Fit a kernel SVM by sequential minimal optimisation.

    The dual problem is

    .. math::
        \max_\alpha \; \sum_i \alpha_i
            - \tfrac12 \sum_{i,j} \alpha_i\alpha_j y_i y_j K(x_i, x_j)
        \quad \text{s.t.} \quad
        0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0,

    and the decision function is
    :math:`f(x) = \sum_i \alpha_i y_i K(x_i, x) + b`.

    Only the kernel enters, never the feature map -- the point of ESL Sec
    12.3. Observations with :math:`\alpha_i = 0` do not appear in ``f`` at
    all, which is what makes the solution sparse in the training set.

    ``C`` trades margin width against violations: small ``C`` gives a wide
    margin and many support vectors, large ``C`` a narrow one and few.

    Parameters
    ----------
    X : array-like
        Training predictors ``(n, p)``.
    y : array-like
        Labels; any two distinct values, mapped internally to -1/+1.
    C : float
        Box constraint, positive.
    kernel : {"rbf", "linear", "poly", "sigmoid"}
        Kernel function.
    gamma : float, optional
        Kernel width for ``"rbf"``/``"sigmoid"``. Defaults to ``1/p``.
    degree, coef0 : float
        Polynomial/sigmoid kernel parameters.
    newdata : array-like, optional
        Points to classify. Defaults to ``X``.
    tol, max_passes, seed
        SMO controls.

    Returns
    -------
    RichResult
        ``alpha``, ``b``, ``support_`` (indices), ``n_support``,
        ``decision`` and ``class_`` at the evaluation points, ``accuracy``
        on training data, and ``dual_gap_check`` reporting
        :math:`\sum_i\alpha_i y_i`, which the constraint forces to zero.

    References
    ----------
    Platt, J. (1998). Sequential minimal optimization. *Microsoft Research
        Technical Report* MSR-TR-98-14.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Two separated Gaussian clouds, classified perfectly.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(-2, 1, (60, 2)), rng.normal(2, 1, (60, 2))]
    >>> y = np.r_[-np.ones(60), np.ones(60)]
    >>> r = esl_svm_kernel(X, y, C=1.0, kernel="rbf", seed=1)
    >>> bool(r["accuracy"] == 1.0)
    True

    The equality constraint holds to machine precision, and no multiplier
    exceeds ``C`` -- the two things a broken SMO gets wrong.

    >>> bool(abs(r["dual_gap_check"]) < 1e-9)
    True
    >>> bool(r["alpha"].max() <= 1.0 + 1e-9)
    True

    A checkerboard is not linearly separable, so a linear kernel fails on it
    where an RBF succeeds -- the reason to have kernels at all.

    >>> Z = rng.uniform(-1, 1, (300, 2))
    >>> yz = np.sign(Z[:, 0] * Z[:, 1])
    >>> lin = esl_svm_kernel(Z, yz, C=1.0, kernel="linear", seed=1)["accuracy"]
    >>> rbf = esl_svm_kernel(Z, yz, C=10.0, kernel="rbf", gamma=2.0, seed=1)["accuracy"]
    >>> bool(lin < 0.7 < rbf)
    True

    >>> esl_svm_kernel(X, y, C=0.0)
    Traceback (most recent call last):
        ...
    ValueError: C must be positive
    """
    if C <= 0:
        raise ValueError("C must be positive")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    yr = np.asarray(y).ravel()
    if X.shape[0] != yr.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {yr.size}")
    classes = np.unique(yr)
    if classes.size != 2:
        raise ValueError(f"y must have exactly 2 classes, found {classes.size}")
    ypm = np.where(yr == classes[1], 1.0, -1.0)

    kw = dict(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    K = kernel_matrix(X, **kw)
    alpha, b, n_iter, converged = smo(K, ypm, C=C, tol=tol, max_passes=max_passes, seed=seed)

    sv = np.flatnonzero(alpha > 1e-8)
    Z = X if newdata is None else np.atleast_2d(np.asarray(newdata, dtype=float))
    dec = kernel_matrix(Z, X, **kw) @ (alpha * ypm) + b
    cls = np.where(dec >= 0, classes[1], classes[0])
    train_dec = K @ (alpha * ypm) + b
    acc = float(np.mean(np.sign(train_dec) == ypm))

    return RichResult(
        title=f"Kernel SVM ({kernel})",
        summary_lines=[("n", int(X.shape[0])), ("support vectors", int(sv.size)),
                       ("C", float(C)), ("train accuracy", acc)],
        warnings=[] if converged else [f"SMO stopped after {n_iter} iterations without settling"],
        payload={
            "alpha": alpha, "b": b, "support_": sv, "n_support": int(sv.size),
            "decision": dec, "class_": cls, "accuracy": acc,
            "dual_gap_check": float(alpha @ ypm),
            "kernel": kernel, "C": float(C), "classes": classes,
            "n_iter": int(n_iter), "converged": bool(converged),
            "method": "esl_svm_kernel",
        },
    )


def cheatsheet():
    return "eslsvm: SMO dual SVM, any of 4 kernels; check dual_gap_check ~ 0 and alpha <= C"


# compact alias per ledger/NAMING.md
eslsvmkernel = esl_svm_kernel
