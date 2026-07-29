# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SGD classifier with hinge loss (linear SVM) trained by stochastic gradient descent."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_sgd_classifier"]


def geron_sgd_classifier(X, y, lr=0.1, n_iter=10, alpha=0.0001, seed=0, shuffle=True):
    """
    SGD classifier with hinge loss (linear SVM) trained by stochastic gradient descent.

    Formula: minimize sum max(0, 1 - y_i f(x_i)) + alpha/2 ||w||^2

    The subgradient of the hinge at sample i is ``-y_i x_i`` while the
    margin is violated (``y_i f(x_i) < 1``) and zero otherwise, so the
    update is

    * violated: ``w <- w - lr*(alpha*w - y_i x_i)``, ``b <- b + lr*y_i``
    * satisfied: ``w <- w - lr*alpha*w`` (regularisation only).

    Labels may be given as ``{-1, +1}`` or ``{0, 1}``; the second is
    mapped to the first, because the hinge is defined on signed margins.
    Sample order is drawn from a deterministic LCG, so a run is
    reproducible without touching global RNG state.

    Parameters
    ----------
    X : array-like
        Design matrix (n, d).
    y : array-like
        Binary labels, exactly two distinct values.
    lr : float, default 0.1
        Learning rate (> 0).
    n_iter : int, default 10
        Epochs over the data (>= 1).
    alpha : float, default 1e-4
        L2 penalty (>= 0).
    seed : int, default 0
        LCG seed for the shuffle.
    shuffle : bool, default True
        Shuffle each epoch; False keeps the given order.

    Returns
    -------
    result : RichResult
        Keys: w, b, loss_curve, decision, predicted, accuracy, n_support,
        estimate, n, method.

    Examples
    --------
    One sample, one epoch, no shuffle: the margin starts at 0 < 1, so the
    first update is exactly ``lr * y * x``:

    >>> r = geron_sgd_classifier([[1.0, 0.0]], [1], lr=0.1, n_iter=1, alpha=0.0, shuffle=False)
    >>> [round(float(v), 12) for v in r["w"]]
    [0.1, 0.0]
    >>> round(float(r["b"]), 12)
    0.1

    A separable problem is solved and the hinge loss falls:

    >>> X = [[2.0, 2.0], [2.0, 1.5], [-2.0, -2.0], [-2.0, -1.5]]
    >>> r2 = geron_sgd_classifier(X, [1, 1, 0, 0], lr=0.1, n_iter=50)
    >>> float(r2["accuracy"])
    1.0
    >>> bool(r2["loss_curve"][-1] < r2["loss_curve"][0])
    True

    References
    ----------
    Géron Ch 3
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.ndim != 2 or Xa.size == 0:
        raise ValueError("geron_sgd_classifier: X must be a non-empty (n, d) design matrix")
    ya = np.asarray(y).ravel()
    if ya.size != Xa.shape[0]:
        raise ValueError(f"geron_sgd_classifier: X has {Xa.shape[0]} rows but y has {ya.size} labels")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("geron_sgd_classifier: X contains non-finite values")
    classes = np.unique(ya)
    if classes.size == 1 and Xa.shape[0] == 1:
        classes = np.asarray([classes[0]])
    elif classes.size != 2:
        raise ValueError(
            f"geron_sgd_classifier: hinge loss needs exactly 2 classes, got {classes.size} ({classes.tolist()})"
        )
    if set(np.unique(ya).tolist()) <= {-1, 1}:
        t = ya.astype(float)
    else:
        pos = classes[-1]
        t = np.where(ya == pos, 1.0, -1.0)
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_sgd_classifier: lr must be positive and finite, got {step}")
    E = int(n_iter)
    if E < 1:
        raise ValueError(f"geron_sgd_classifier: n_iter must be >= 1, got {E}")
    reg = float(alpha)
    if not np.isfinite(reg) or reg < 0:
        raise ValueError(f"geron_sgd_classifier: alpha must be non-negative and finite, got {reg}")

    n, d = Xa.shape
    w = np.zeros(d)
    b = 0.0
    rng = int(seed) % 2**32

    def _u():
        nonlocal rng
        rng = (1664525 * rng + 1013904223) % 2**32
        return (rng + 0.5) / 2**32

    losses = []
    for _ in range(E):
        order = list(range(n))
        if shuffle:
            for i in range(n - 1, 0, -1):  # Fisher-Yates on the LCG stream
                j = int(_u() * (i + 1))
                order[i], order[j] = order[j], order[i]
        for i in order:
            margin = t[i] * (Xa[i] @ w + b)
            if margin < 1.0:
                w = w - step * (reg * w - t[i] * Xa[i])
                b = b + step * t[i]
            else:
                w = w - step * reg * w
        f = Xa @ w + b
        losses.append(float(np.mean(np.maximum(0.0, 1.0 - t * f)) + 0.5 * reg * float(w @ w)))

    f = Xa @ w + b
    pred_pm = np.where(f >= 0, 1.0, -1.0)
    pred = np.where(pred_pm > 0, classes[-1], classes[0]) if classes.size == 2 else pred_pm
    acc = float(np.mean(pred_pm == t))

    return RichResult(
        title="Linear SVM by SGD (hinge loss)",
        summary_lines=[("Samples", n), ("Features", d), ("Epochs", E), ("Training accuracy", acc)],
        interpretation=(
            "Only margin violators move the weights, so once the data are separated with margin the "
            "updates are pure weight decay -- the support vectors are the samples still inside the margin."
        ),
        payload={
            "w": w,
            "b": float(b),
            "loss_curve": np.asarray(losses, dtype=float),
            "decision": f,
            "predicted": pred,
            "accuracy": acc,
            "n_support": int(np.sum(t * f < 1.0)),
            "classes": classes,
            "estimate": float(losses[-1]),
            "n": int(n),
            "method": "Hinge-loss linear SVM trained by stochastic subgradient descent with L2 decay",
        },
    )


def cheatsheet():
    return "hmsgdc: SGD classifier with hinge loss (linear SVM) trained by stochastic gradient descent"
