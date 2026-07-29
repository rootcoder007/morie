# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-entropy cost for K-class softmax regression."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_cross_entropy_cost"]


def _softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def geron_cross_entropy_cost(X, Y, theta):
    """
    Cross-entropy cost for K-class softmax regression.

    Formula: J = -(1/m) sum_i sum_k y_ik log p_ik

    The scores ``X @ theta`` are turned into probabilities by a
    max-shifted softmax, so the cost is finite even for large logits. The
    log is evaluated on the softmax output computed in a numerically
    stable way (``log p = s - logsumexp(s)``), which means a confidently
    wrong prediction produces a large finite cost rather than ``inf``.

    ``Y`` may be one-hot ``(m, K)`` or a vector of ``m`` class indices.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix (include a bias column yourself if you want one).
    Y : array-like, shape (m, K) or (m,)
        One-hot targets or integer class labels.
    theta : array-like, shape (n, K)
        Parameter matrix, one column per class.

    Returns
    -------
    result : RichResult
        Keys: cost, probabilities, log_probabilities, per_sample_cost,
        accuracy, chance_cost, estimate, n, method.

    Examples
    --------
    Zero parameters make every class equally likely, so the cost is
    ``log K``:

    >>> import math
    >>> r = geron_cross_entropy_cost([[1.0, 2.0], [3.0, 4.0]], [0, 1],
    ...                              [[0.0, 0.0], [0.0, 0.0]])
    >>> round(r["cost"], 9) == round(math.log(2), 9)
    True
    >>> [round(p, 6) for p in r["probabilities"][0]]
    [0.5, 0.5]

    A confident correct prediction costs almost nothing; flipping the
    label costs the full logit gap:

    >>> r2 = geron_cross_entropy_cost([[1.0]], [[1.0, 0.0]], [[10.0, 0.0]])
    >>> f"{r2['cost']:.6f}"
    '0.000045'
    >>> r3 = geron_cross_entropy_cost([[1.0]], [[0.0, 1.0]], [[10.0, 0.0]])
    >>> f"{r3['cost']:.6f}"
    '10.000045'

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    th = np.atleast_2d(np.asarray(theta, dtype=float))
    if X.size == 0:
        raise ValueError("geron_cross_entropy_cost: X is empty")
    if X.shape[1] != th.shape[0]:
        raise ValueError(f"geron_cross_entropy_cost: X has {X.shape[1]} columns but theta has {th.shape[0]} rows")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(th)):
        raise ValueError("geron_cross_entropy_cost: X and theta must be finite")
    m, K = X.shape[0], th.shape[1]

    Yarr = np.asarray(Y)
    if Yarr.ndim == 1 or (Yarr.ndim == 2 and 1 in Yarr.shape and Yarr.size == m and K != 1):
        idx = Yarr.ravel()
        if idx.size != m:
            raise ValueError(f"geron_cross_entropy_cost: Y has {idx.size} labels but X has {m} rows")
        ii = idx.astype(int)
        if not np.array_equal(ii, idx):
            raise ValueError("geron_cross_entropy_cost: class labels must be whole numbers")
        if ii.min() < 0 or ii.max() >= K:
            raise ValueError(f"geron_cross_entropy_cost: labels must lie in 0..{K - 1}, got {ii.min()}..{ii.max()}")
        Yoh = np.zeros((m, K))
        Yoh[np.arange(m), ii] = 1.0
    else:
        Yoh = np.asarray(Yarr, dtype=float)
        if Yoh.shape != (m, K):
            raise ValueError(f"geron_cross_entropy_cost: Y must have shape {(m, K)}, got {Yoh.shape}")
        if np.any(Yoh < 0):
            raise ValueError("geron_cross_entropy_cost: target probabilities must be non-negative")
        rows = Yoh.sum(axis=1)
        if not np.allclose(rows, 1.0):
            raise ValueError("geron_cross_entropy_cost: each row of Y must sum to 1")

    S = X @ th
    Sm = S - S.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(Sm).sum(axis=1, keepdims=True))
    logp = Sm - logZ
    P = np.exp(logp)
    per = -np.sum(Yoh * logp, axis=1)
    cost = float(np.mean(per))
    acc = float(np.mean(P.argmax(axis=1) == Yoh.argmax(axis=1)))

    return RichResult(
        title="Softmax cross-entropy cost",
        summary_lines=[("Cost", cost), ("Classes", int(K)), ("Accuracy", acc)],
        interpretation=f"Chance cost for {K} classes is log K = {float(np.log(K)):.6f}.",
        payload={
            "cost": cost,
            "probabilities": P.tolist(),
            "log_probabilities": logp.tolist(),
            "per_sample_cost": per.tolist(),
            "scores": S.tolist(),
            "accuracy": acc,
            "chance_cost": float(np.log(K)),
            "n_classes": int(K),
            "estimate": cost,
            "n": int(m),
            "method": "softmax cross-entropy J = -(1/m) sum_i sum_k y_ik log p_ik",
        },
    )


def cheatsheet():
    return "hmcec: Cross-entropy cost for K-class softmax regression"
