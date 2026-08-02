# morie.fn -- function file (rootcoder007/morie)
"""Backpropagation -- ESL Sec 11.4."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_backprop"]


def esl_backprop(X, y, weights, task="regression"):
    r"""One backpropagation sweep: forward pass, then the gradients.

    The backward recurrence of ESL eq. (11.5), for the single-hidden-layer
    network:

    .. math::
        \delta_{ik} = \frac{\partial L}{\partial T_k}, \qquad
        s_{im} = \sigma'(a_{im}) \sum_k \beta_{km}\,\delta_{ik},

    giving :math:`\partial L/\partial \beta_{km} = \sum_i \delta_{ik} z_{im}`
    and :math:`\partial L/\partial \alpha_{m\ell} = \sum_i s_{im} x_{i\ell}`.

    The content of backprop is the chain rule *reusing* :math:`\delta`: the
    output error is computed once and propagated, instead of being
    recomputed for each weight. That is what turns an O(#weights) forward-
    difference cost into a single extra sweep.

    ``weights`` is the mapping used by
    :func:`~morie.fn.eslnnt.esl_neural_net` -- keys ``alpha``, ``alpha0``,
    ``beta``, ``beta0``. Returning the gradients rather than applying them
    keeps the choice of optimiser (see :func:`~morie.fn.adamopt.adam`)
    separate from the derivative.

    Parameters
    ----------
    X : array-like
        Inputs ``(n, p)``, already scaled as the network expects.
    y : array-like
        Targets. Numeric for regression, 0-based class indices otherwise.
    weights : dict
        ``alpha`` ``(p, M)``, ``alpha0`` ``(M,)``, ``beta`` ``(M, K)``,
        ``beta0`` ``(K,)``.
    task : {"regression", "classification"}
        Sets the output layer and loss.

    Returns
    -------
    RichResult
        ``grad_alpha``, ``grad_alpha0``, ``grad_beta``, ``grad_beta0``,
        ``delta``, ``hidden``, ``loss``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Backprop agrees with a central finite difference to 8 decimals -- the
    only test that actually validates a gradient.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 3))
    >>> y = rng.normal(size=40)
    >>> W = {"alpha": rng.normal(size=(3, 4)) * 0.5, "alpha0": np.zeros(4),
    ...      "beta": rng.normal(size=(4, 1)) * 0.5, "beta0": np.zeros(1)}
    >>> g = esl_backprop(X, y, W)["grad_alpha"]
    >>> def loss(W):
    ...     Z = 1 / (1 + np.exp(-(X @ W["alpha"] + W["alpha0"])))
    ...     return float(np.mean((Z @ W["beta"] + W["beta0"] - y[:, None]) ** 2))
    >>> h, num = 1e-6, np.zeros_like(g)
    >>> for i in range(3):
    ...     for j in range(4):
    ...         Wp = {k: v.copy() for k, v in W.items()}; Wp["alpha"][i, j] += h
    ...         Wm = {k: v.copy() for k, v in W.items()}; Wm["alpha"][i, j] -= h
    ...         num[i, j] = (loss(Wp) - loss(Wm)) / (2 * h)
    >>> bool(np.max(np.abs(g - num)) < 1e-8)
    True

    A gradient step reduces the loss.

    >>> r = esl_backprop(X, y, W)
    >>> W2 = {"alpha": W["alpha"] - 0.1 * r["grad_alpha"],
    ...       "alpha0": W["alpha0"] - 0.1 * r["grad_alpha0"],
    ...       "beta": W["beta"] - 0.1 * r["grad_beta"],
    ...       "beta0": W["beta0"] - 0.1 * r["grad_beta0"]}
    >>> bool(esl_backprop(X, y, W2)["loss"] < r["loss"])
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    for key in ("alpha", "alpha0", "beta", "beta0"):
        if key not in weights:
            raise ValueError(f"weights is missing {key!r}")
    a = np.asarray(weights["alpha"], dtype=float)
    a0 = np.asarray(weights["alpha0"], dtype=float).ravel()
    b = np.atleast_2d(np.asarray(weights["beta"], dtype=float))
    b0 = np.asarray(weights["beta0"], dtype=float).ravel()
    if a.shape[0] != p:
        raise ValueError(f"alpha has {a.shape[0]} rows but X has {p} columns")
    M, K = b.shape
    if a.shape[1] != M:
        raise ValueError(f"alpha has {a.shape[1]} hidden units but beta has {M}")

    A = X @ a + a0
    Z = 1.0 / (1.0 + np.exp(-np.clip(A, -500, 500)))
    T = Z @ b + b0

    yr = np.asarray(y).ravel()
    if task == "regression":
        Y = yr.astype(float).reshape(n, K)
        delta = 2 * (T - Y) / n
        loss = float(np.mean((T - Y) ** 2))
    elif task == "classification":
        Y = np.zeros((n, K))
        Y[np.arange(n), yr.astype(int)] = 1.0
        e = np.exp(T - T.max(axis=1, keepdims=True))
        P = e / e.sum(axis=1, keepdims=True)
        delta = (P - Y) / n
        loss = float(-np.mean(np.sum(Y * np.log(P + 1e-300), axis=1)))
    else:
        raise ValueError('task must be "regression" or "classification"')

    s = (delta @ b.T) * Z * (1 - Z)          # ESL eq. (11.5)
    return RichResult(
        title="Backpropagation sweep",
        summary_lines=[("n", n), ("hidden units", int(M)), ("loss", loss)],
        payload={
            "grad_alpha": X.T @ s, "grad_alpha0": s.sum(axis=0),
            "grad_beta": Z.T @ delta, "grad_beta0": delta.sum(axis=0),
            "delta": delta, "s": s, "hidden": Z, "output": T, "loss": loss,
            "method": "esl_backprop",
        },
    )


def cheatsheet():
    return "eslbpr: returns gradients, not a step; validated against central differences to 1e-8"
