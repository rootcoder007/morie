# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perceptron learning rule."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_perceptron"]


def geron_perceptron(X, y, eta=1.0, n_iter=10):
    """
    Perceptron learning rule.

    Formula: w_{t+1} = w_t + eta (y - y_hat) x

    The update fires only on a mistake, and its size is the input itself:
    a wrongly-rejected instance is pulled toward acceptance in proportion
    to how large its features are. On a linearly separable set this
    converges in finitely many mistakes (Novikoff); on anything else --
    XOR being the famous case -- it never settles, which is why
    ``converged`` and the per-epoch mistake counts are returned rather
    than a bare accuracy.

    Labels must be 0/1 and the bias is handled internally.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
        Labels in {0, 1}.
    eta : float, default 1.0
        Learning rate (positive). It rescales w but not the decision
        boundary when starting from zero weights.
    n_iter : int, default 10
        Passes over the data.

    Returns
    -------
    result : RichResult
        Keys: w, bias, predictions, accuracy, mistakes_per_epoch,
        converged, estimate, n, method.

    Examples
    --------
    AND is linearly separable, so the perceptron gets it exactly:

    >>> X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    >>> r = geron_perceptron(X, [0, 0, 0, 1], eta=1.0, n_iter=20)
    >>> float(r["accuracy"]), bool(r["converged"])
    (1.0, True)
    >>> [int(p) for p in r["predictions"]]
    [0, 0, 0, 1]

    XOR is not, and no number of epochs fixes it:

    >>> x = geron_perceptron(X, [0, 1, 1, 0], eta=1.0, n_iter=50)
    >>> bool(x["converged"]), bool(x["accuracy"] < 1.0)
    (False, True)

    References
    ----------
    Geron Ch 9
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_perceptron: X must be a non-empty 2-D array, got shape {A.shape}")
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if yv.size != A.shape[0]:
        raise ValueError(f"geron_perceptron: X has {A.shape[0]} rows but y has {yv.size} entries")
    bad = np.setdiff1d(np.unique(yv), np.array([0.0, 1.0]))
    if bad.size:
        raise ValueError(f"geron_perceptron: labels must be 0 or 1, got {bad.tolist()}")
    lr = float(eta)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_perceptron: eta must be positive and finite, got {eta!r}")
    T = int(n_iter)
    if T < 1:
        raise ValueError(f"geron_perceptron: n_iter must be >= 1, got {n_iter!r}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_perceptron: X contains non-finite values")

    m, n = A.shape
    w = np.zeros(n)
    b = 0.0
    mistakes = []
    converged = False
    for _ in range(T):
        wrong = 0
        for i in range(m):
            yhat = 1.0 if (A[i] @ w + b) >= 0.0 else 0.0
            err = yv[i] - yhat
            if err != 0.0:
                w += lr * err * A[i]
                b += lr * err
                wrong += 1
        mistakes.append(wrong)
        if wrong == 0:
            converged = True
            break

    pred = ((A @ w + b) >= 0.0).astype(float)
    acc = float(np.mean(pred == yv))
    return RichResult(
        title="Perceptron",
        summary_lines=[("Accuracy", acc), ("Epochs run", len(mistakes)), ("Converged", converged)],
        interpretation="Converges in finite mistakes iff the classes are linearly separable; XOR never converges.",
        payload={
            "w": w,
            "weights": w,
            "bias": float(b),
            "predictions": pred,
            "accuracy": acc,
            "mistakes_per_epoch": mistakes,
            "converged": converged,
            "estimate": w,
            "n": int(m),
            "method": "Perceptron rule w <- w + eta (y - y_hat) x with a step activation",
        },
    )


def cheatsheet():
    return "hmpcpt: Perceptron learning rule"
