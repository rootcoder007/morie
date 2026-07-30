# morie.fn -- function file (rootcoder007/morie)
"""Single-hidden-layer neural network -- ESL Sec 11.3-11.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_neural_net"]


def esl_neural_net(X, y, M=5, lambda_=0.0, lr=0.1, n_epochs=400, task="regression",
                   newdata=None, seed=0, standardize=True):
    r"""Fit ESL's single-hidden-layer network by gradient descent.

    The architecture of Sec 11.3, with ``M`` hidden units:

    .. math::
        Z_m = \sigma(\alpha_{0m} + \alpha_m^\top X), \qquad
        T_k = \beta_{0k} + \beta_k^\top Z, \qquad
        f_k(X) = g_k(T),

    with :math:`\sigma` the sigmoid and :math:`g` the identity for regression
    or softmax for classification.

    Two points ESL is emphatic about, both honoured here. Weights are
    initialised **small but non-zero**: exactly zero leaves the model
    perfectly symmetric, every hidden unit computes the same thing and stays
    that way, while large starting weights saturate the sigmoid and kill the
    gradient. And inputs are standardised by default, because the weight
    decay penalty is otherwise a statement about the predictors' measurement
    units.

    ``lambda_`` penalises all weights except the biases -- the usual
    convention, since shrinking an intercept toward zero is a claim about the
    origin.

    Parameters
    ----------
    X : array-like
        Predictors ``(n, p)``.
    y : array-like
        Response. Numeric for regression, class labels for classification.
    M : int
        Hidden units, at least 1.
    lambda_ : float
        Weight decay, non-negative.
    lr : float
        Learning rate.
    n_epochs : int
        Full-batch gradient steps.
    task : {"regression", "classification"}
        Output layer and loss.
    newdata : array-like, optional
        Points to predict at. Defaults to ``X``.
    seed : int
        Seed for weight initialisation.
    standardize : bool
        Standardise the inputs.

    Returns
    -------
    RichResult
        ``fitted`` (or ``prob``/``class_``), ``alpha``, ``beta``,
        ``loss_path``, ``hidden`` activations, ``r_squared`` or ``accuracy``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    A non-linear regression surface a linear model cannot represent.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-2, 2, (400, 2))
    >>> y = np.sin(X[:, 0]) + X[:, 1] ** 2
    >>> r = esl_neural_net(X, y, M=8, lr=0.3, n_epochs=3000, seed=1)
    >>> bool(r["r_squared"] > 0.9)
    True

    The loss decreases monotonically under full-batch gradient descent.

    >>> p = np.asarray(r["loss_path"])
    >>> bool(p[-1] < p[0] and np.all(np.diff(p) < 1e-9))
    True

    Classification on separable classes.

    >>> Z = np.r_[rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))]
    >>> yz = np.r_[np.zeros(100), np.ones(100)]
    >>> c = esl_neural_net(Z, yz, M=4, task="classification", lr=0.5,
    ...                    n_epochs=800, seed=1)
    >>> bool(c["accuracy"] > 0.95)
    True

    >>> esl_neural_net(X, y, M=0)
    Traceback (most recent call last):
        ...
    ValueError: M must be at least 1
    """
    if M < 1:
        raise ValueError("M must be at least 1")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    yr = np.asarray(y).ravel()
    n, p = X.shape
    if yr.size != n:
        raise ValueError(f"X has {n} rows but y has {yr.size}")

    mu = X.mean(axis=0) if standardize else np.zeros(p)
    sd = X.std(axis=0, ddof=0) if standardize else np.ones(p)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd

    if task == "classification":
        classes = np.unique(yr)
        K = classes.size
        Y = np.zeros((n, K))
        Y[np.arange(n), np.searchsorted(classes, yr)] = 1.0
    elif task == "regression":
        classes, K = None, 1
        Y = yr.astype(float).reshape(n, 1)
    else:
        raise ValueError('task must be "regression" or "classification"')

    rng = np.random.default_rng(seed)
    # Small but non-zero: zero is a symmetric saddle, large saturates sigma.
    a = rng.uniform(-0.7, 0.7, (p, M))
    a0 = np.zeros(M)
    b = rng.uniform(-0.7, 0.7, (M, K))
    b0 = np.zeros(K)

    losses = []
    for _ in range(n_epochs):
        Zh = 1.0 / (1.0 + np.exp(-np.clip(Xs @ a + a0, -500, 500)))
        T = Zh @ b + b0
        if task == "regression":
            P = T
            err = P - Y
            loss = float(np.mean(err**2))
        else:
            e = np.exp(T - T.max(axis=1, keepdims=True))
            P = e / e.sum(axis=1, keepdims=True)
            err = P - Y
            loss = float(-np.mean(np.sum(Y * np.log(P + 1e-300), axis=1)))
        loss += lambda_ * (float(np.sum(a**2)) + float(np.sum(b**2)))
        losses.append(loss)

        gT = 2 * err / n if task == "regression" else err / n
        gb = Zh.T @ gT + 2 * lambda_ * b
        gb0 = gT.sum(axis=0)
        gZ = gT @ b.T * Zh * (1 - Zh)
        ga = Xs.T @ gZ + 2 * lambda_ * a
        ga0 = gZ.sum(axis=0)
        a -= lr * ga
        a0 -= lr * ga0
        b -= lr * gb
        b0 -= lr * gb0

    Zt = X if newdata is None else np.atleast_2d(np.asarray(newdata, dtype=float))
    if Zt.shape[1] != p:
        raise ValueError(f"newdata has {Zt.shape[1]} columns but X has {p}")
    Hh = 1.0 / (1.0 + np.exp(-np.clip(((Zt - mu) / sd) @ a + a0, -500, 500)))
    T = Hh @ b + b0

    Htr = 1.0 / (1.0 + np.exp(-np.clip(Xs @ a + a0, -500, 500)))
    Ttr = Htr @ b + b0
    payload = {
        "alpha": a, "alpha0": a0, "beta": b, "beta0": b0,
        "hidden": Hh, "loss_path": np.array(losses),
        "M": int(M), "lambda_": float(lambda_), "task": task,
        "mean": mu, "sd": sd,
        "method": "esl_neural_net",
    }
    if task == "regression":
        fit_tr = Ttr.ravel()
        ss = float(np.sum((yr - yr.mean()) ** 2))
        payload.update({
            "fitted": T.ravel(),
            "r_squared": float(1 - np.sum((yr - fit_tr) ** 2) / ss) if ss > 0 else np.nan,
        })
        head = [("R^2", payload["r_squared"])]
    else:
        e = np.exp(T - T.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        etr = np.exp(Ttr - Ttr.max(axis=1, keepdims=True))
        ptr = etr / etr.sum(axis=1, keepdims=True)
        acc = float(np.mean(classes[ptr.argmax(1)] == yr))
        payload.update({
            "prob": prob, "class_": classes[prob.argmax(1)],
            "classes": classes, "accuracy": acc,
        })
        head = [("accuracy", acc)]
    return RichResult(
        title=f"Neural network ({task}, M={M})",
        summary_lines=[("n", n), ("p", p), ("M", int(M))] + head,
        payload=payload,
    )


def cheatsheet():
    return "eslnnt: 1 hidden layer; init small but NON-ZERO (zero is a symmetric saddle), standardise inputs"
