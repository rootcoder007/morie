# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rosenblatt perceptron (ESL Ch 4.5.1)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_perceptron"]


def esl_perceptron(X, y, eta=1.0, max_epoch=1000):
    """
    Perceptron: beta <- beta + eta y_i x_i whenever y_i(beta'x_i) <= 0.

    Cycles the observations in FIXED order, updating on every
    misclassification. The convergence guarantee is one-directional
    and worth stating: on separable data the perceptron reaches zero
    errors in finite time, but on NON-separable data it never
    converges and simply cycles — so the loop is capped and
    ``converged`` is reported honestly rather than the final beta
    being presented as a solution. ESL Ch 4.5.1 also notes the
    solution is not unique and depends on the starting values and the
    order of the data, which is why the order here is fixed and
    documented rather than shuffled.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Features; include your own bias column if you want an offset.
    y : array-like, shape (n,)
        Labels in {-1, +1}.
    eta : float
        Learning rate, > 0.
    max_epoch : int
        Passes over the data before giving up.

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, n_errors, epochs,
        converged, separable_within_budget, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 4.5.1;
    Rosenblatt (1958).

    Examples
    --------
    Separable data converges to zero errors:

    >>> X = [[1.0, 2.0], [1.0, 3.0], [1.0, -2.0], [1.0, -3.0]]
    >>> y = [1, 1, -1, -1]
    >>> out = esl_perceptron(X, y)
    >>> out["converged"]
    True
    >>> out["n_errors"]
    0

    Non-separable data does not, and says so:

    >>> bad = esl_perceptron([[1.0, 0.0], [1.0, 0.0]], [1, -1], max_epoch=50)
    >>> bad["converged"]
    False
    >>> bad["n_errors"] > 0
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    eta = float(eta)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} labels.")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("labels must lie in {-1, +1}.")
    if eta <= 0:
        raise ValueError(f"the learning rate must be positive; got {eta}.")
    beta = np.zeros(p)
    converged = False
    ep = 0
    for ep in range(1, int(max_epoch) + 1):
        errors = 0
        for i in range(n):
            if y[i] * float(X[i] @ beta) <= 0:
                beta = beta + eta * y[i] * X[i]
                errors += 1
        if errors == 0:
            converged = True
            break
    final_err = int(np.sum(y * (X @ beta) <= 0))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "n_errors": final_err, "epochs": int(ep), "converged": bool(converged),
        "separable_within_budget": bool(converged), "n": int(n), "p": int(p),
        "method": "Rosenblatt perceptron, fixed data order; no convergence if not separable"})


def cheatsheet():
    return "eslprc: update on misclassification; cycles forever if not separable"
