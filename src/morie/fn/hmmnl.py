# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multinomial logistic (softmax) regression end-to-end fit."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_multinomial_logistic"]

_METHOD = "Softmax regression fitted by batch gradient descent"


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def geron_multinomial_logistic(X, Y, lr=0.1, n_iter=1000, add_bias=True, alpha=0.0):
    """
    Multinomial logistic (softmax) regression end-to-end fit.

    Formula: theta* = argmin CrossEntropy(softmax(X Theta), Y)

    Softmax regression fitted by full-batch gradient descent.  The
    gradient of the cross-entropy with respect to the logits is exactly
    ``P - Y``, so the parameter gradient is ``(1/m) X^T (P - Y)``: the
    same shape as binary logistic regression, one column per class.
    That cancellation is the reason softmax and cross-entropy are used
    together and not separately.

    Softmax is shift-invariant -- adding a constant to every logit of a
    row changes nothing -- so ``Theta`` is identified only up to a
    per-feature constant across classes, and the unregularised solution
    is not unique.  Probabilities and predictions are, which is why they,
    not the coefficients, are what should be compared between runs.  A
    positive ``alpha`` (L2 on the non-bias rows) restores uniqueness.

    The logits are shifted by their row maximum before exponentiating,
    so a logit of 1000 saturates rather than returning ``inf/inf``.

    ``Y`` is either a 1-D vector of class indices or a one-hot matrix.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Features.
    Y : array-like, shape (m,) or (m, K)
        Class indices or one-hot labels.
    lr : float
        Learning rate.
    n_iter : int
        Gradient-descent iterations.
    add_bias : bool
        Prepend a ones column.
    alpha : float
        L2 strength on the non-bias rows.

    Returns
    -------
    result : RichResult
        Keys: Theta, probabilities, prediction, loss_history, loss,
        accuracy, estimate, n, method.

    Examples
    --------
    Three separable classes on a line are learned perfectly:

    >>> X = [[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]]
    >>> y = [0, 0, 1, 1, 2, 2]
    >>> r = geron_multinomial_logistic(X, y, lr=0.5, n_iter=3000)
    >>> float(r["accuracy"])
    1.0
    >>> [int(v) for v in r["prediction"]]
    [0, 0, 1, 1, 2, 2]

    Probabilities are distributions, and the loss falls monotonically
    under full-batch descent:

    >>> [round(float(v), 12) for v in np.sum(r["probabilities"], axis=1)][:2]
    [1.0, 1.0]
    >>> bool(r["loss_history"][-1] < r["loss_history"][0] / 100)
    True

    At initialisation every class is equally likely, so the loss starts
    at ``log 3``:

    >>> z = geron_multinomial_logistic(X, y, lr=0.5, n_iter=1)
    >>> round(float(z["loss_history"][0]), 9)
    1.098612289

    One-hot labels are accepted too:

    >>> onehot = np.eye(3)[y]
    >>> o = geron_multinomial_logistic(X, onehot, lr=0.5, n_iter=3000)
    >>> float(o["accuracy"])
    1.0

    References
    ----------
    Géron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_multinomial_logistic: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_multinomial_logistic: X contains non-finite values")
    m = A.shape[0]

    Ya = np.asarray(Y)
    if Ya.ndim == 1 or (Ya.ndim == 2 and Ya.shape[1] == 1):
        idx = np.asarray(Ya, dtype=float).ravel()
        if idx.size != m:
            raise ValueError(f"geron_multinomial_logistic: X has {m} rows but Y has {idx.size} entries")
        if not np.all(idx == np.floor(idx)) or np.any(idx < 0):
            raise ValueError("geron_multinomial_logistic: class indices must be non-negative integers")
        idx = idx.astype(int)
        K = int(idx.max()) + 1
        if K < 2:
            raise ValueError(
                f"geron_multinomial_logistic: only one class present; softmax regression needs at least 2"
            )
        Yh = np.eye(K)[idx]
    else:
        Yh = np.asarray(Ya, dtype=float)
        if Yh.shape[0] != m:
            raise ValueError(f"geron_multinomial_logistic: X has {m} rows but Y has {Yh.shape[0]}")
        if not np.allclose(Yh.sum(axis=1), 1.0) or np.any(Yh < 0):
            raise ValueError("geron_multinomial_logistic: one-hot Y must have non-negative rows summing to 1")
        K = Yh.shape[1]
        idx = np.argmax(Yh, axis=1)

    if add_bias:
        A = np.hstack([np.ones((m, 1)), A])
    n = A.shape[1]
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_multinomial_logistic: lr must be positive and finite, got {lr!r}")
    iters = int(n_iter)
    if iters < 1:
        raise ValueError(f"geron_multinomial_logistic: n_iter must be at least 1, got {n_iter!r}")
    a = float(alpha)
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"geron_multinomial_logistic: alpha must be non-negative and finite, got {alpha!r}")

    Theta = np.zeros((n, K))
    penalty_mask = np.ones((n, K))
    if add_bias:
        penalty_mask[0, :] = 0.0

    history = []
    for _ in range(iters):
        P = _softmax(A @ Theta)
        loss = float(np.mean(-np.log(np.clip(P[np.arange(m), idx], 1e-300, None))))
        if a > 0:
            loss += 0.5 * a * float(np.sum((Theta * penalty_mask) ** 2))
        if not np.isfinite(loss):
            raise ValueError(f"geron_multinomial_logistic: the loss diverged at lr={eta}; lower it")
        history.append(loss)
        grad = (A.T @ (P - Yh)) / m + a * Theta * penalty_mask
        Theta = Theta - eta * grad

    P = _softmax(A @ Theta)
    final_loss = float(np.mean(-np.log(np.clip(P[np.arange(m), idx], 1e-300, None))))
    pred = np.argmax(P, axis=1)
    acc = float(np.mean(pred == idx))

    return RichResult(
        title="Softmax regression",
        summary_lines=[
            ("Classes", int(K)),
            ("Iterations", iters),
            ("Cross-entropy", final_loss),
            ("Accuracy", acc),
        ],
        warnings=(
            [
                "alpha = 0: softmax is shift-invariant, so Theta is identified only up to a per-feature "
                "constant across classes. Compare probabilities, not coefficients."
            ]
            if a == 0
            else []
        ),
        interpretation=(
            "The logit gradient is exactly P - Y; that cancellation is why softmax and cross-entropy "
            "belong together."
        ),
        payload={
            "Theta": Theta,
            "probabilities": P,
            "prediction": pred,
            "loss_history": np.asarray(history),
            "loss": final_loss,
            "accuracy": acc,
            "n_classes": int(K),
            "estimate": final_loss,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmnl: softmax regression fitted by GD; gradient (1/m) X^T (P - Y), shift-invariance flagged"
