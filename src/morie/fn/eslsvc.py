# morie.fn -- function file (rootcoder007/morie)
"""Linear support vector classifier -- ESL Sec 12.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._svm import kernel_matrix, smo

__all__ = ["esl_svc"]


def esl_svc(X, y, C=1.0, newdata=None, tol=1e-3, max_passes=50, seed=0):
    r"""Soft-margin linear support vector classifier.

    The primal problem of ESL Sec 12.2 is

    .. math::
        \min_{w, b, \xi} \; \tfrac12 \lVert w \rVert^2 + C\sum_i \xi_i
        \quad\text{s.t.}\quad
        y_i(w^\top x_i + b) \ge 1 - \xi_i, \;\; \xi_i \ge 0,

    solved here through its dual, from which the primal weights are recovered
    as :math:`w = \sum_i \alpha_i y_i x_i`. Unlike
    :func:`~morie.fn.eslsvm.esl_svm_kernel` the weight vector is available
    explicitly, so the fitted rule is interpretable coefficient by
    coefficient.

    The margin width is :math:`2/\lVert w \rVert`, and it is reported: a
    classifier that separates the data with a wide margin is a different
    object from one that just separates it.

    Parameters
    ----------
    X : array-like
        Predictors ``(n, p)``.
    y : array-like
        Labels; any two distinct values.
    C : float
        Cost of margin violations, positive. Small ``C`` widens the margin.
    newdata : array-like, optional
        Points to classify. Defaults to ``X``.
    tol, max_passes, seed
        SMO controls.

    Returns
    -------
    RichResult
        ``w``, ``b``, ``margin``, ``alpha``, ``support_``, ``decision``,
        ``class_``, ``accuracy``, ``n_violations``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    On separable data the fitted normal points from the negative class to
    the positive one along the true axis.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(-3, 0.5, (50, 2)), rng.normal(3, 0.5, (50, 2))]
    >>> y = np.r_[-np.ones(50), np.ones(50)]
    >>> r = esl_svc(X, y, C=1.0, seed=1)
    >>> bool(r["accuracy"] == 1.0 and r["w"][0] > 0)
    True

    Lowering ``C`` buys a wider margin -- the trade the formulation makes.

    >>> wide = esl_svc(X, y, C=0.01, seed=1)["margin"]
    >>> narrow = esl_svc(X, y, C=100.0, seed=1)["margin"]
    >>> bool(wide > narrow)
    True

    The linear rule agrees with the linear-kernel SVM, as it must.

    >>> from morie.fn.eslsvm import esl_svm_kernel
    >>> k = esl_svm_kernel(X, y, C=1.0, kernel="linear", seed=1)
    >>> bool(np.array_equal(np.sign(r["decision"]), np.sign(k["decision"])))
    True
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

    K = kernel_matrix(X, kernel="linear")
    alpha, b, n_iter, converged = smo(K, ypm, C=C, tol=tol, max_passes=max_passes, seed=seed)
    w = (alpha * ypm) @ X
    wn = float(np.linalg.norm(w))

    Z = X if newdata is None else np.atleast_2d(np.asarray(newdata, dtype=float))
    if Z.shape[1] != X.shape[1]:
        raise ValueError(f"newdata has {Z.shape[1]} columns but X has {X.shape[1]}")
    dec = Z @ w + b
    train_dec = X @ w + b
    slack = np.maximum(0.0, 1.0 - ypm * train_dec)

    return RichResult(
        title="Linear SVC (soft margin)",
        summary_lines=[("n", int(X.shape[0])), ("C", float(C)),
                       ("margin", 2.0 / wn if wn > 0 else np.inf),
                       ("support vectors", int((alpha > 1e-8).sum()))],
        warnings=[] if converged else [f"SMO stopped after {n_iter} iterations without settling"],
        payload={
            "w": w, "b": b,
            "margin": 2.0 / wn if wn > 0 else float("inf"),
            "w_norm": wn,
            "alpha": alpha, "support_": np.flatnonzero(alpha > 1e-8),
            "decision": dec, "class_": np.where(dec >= 0, classes[1], classes[0]),
            "accuracy": float(np.mean(np.sign(train_dec) == ypm)),
            "slack": slack, "n_violations": int(np.sum(slack > 1e-8)),
            "classes": classes, "C": float(C),
            "n_iter": int(n_iter), "converged": bool(converged),
            "method": "esl_svc",
        },
    )


def cheatsheet():
    return "eslsvc: linear soft-margin SVC; w is explicit and margin = 2/||w|| widens as C falls"
