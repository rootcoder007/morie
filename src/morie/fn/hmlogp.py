# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logistic regression probability prediction."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_logistic_probability", "sigmoid", "add_bias_column"]

_METHOD = "Logistic regression probability"


def sigmoid(z):
    """Overflow-safe logistic function, shared by the logistic family.

    ``1/(1+exp(-z))`` overflows ``exp`` for z around -710 and warns long
    before that; splitting on the sign keeps every ``exp`` argument
    non-positive.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def add_bias_column(X):
    """Prepend a column of ones to a design matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def geron_logistic_probability(X, theta, add_bias=False):
    """
    Logistic regression probability prediction.

    Formula: p_hat = sigma(theta^T x)

    The decision boundary sits at ``p = 0.5``, i.e. exactly where
    ``theta^T x = 0``, so the boundary is linear in ``x`` even though
    the probability is not.  The logits are returned next to the
    probabilities because they are what the boundary is defined on and
    what the gradient in :func:`geron_logistic_gradient` differentiates.

    The sigmoid here is evaluated by sign-split so that a logit of -800
    returns 0.0 rather than raising an overflow warning and an ``inf``.

    Parameters
    ----------
    X : array-like, shape (m, n) or (n,)
        Design matrix; a 1-D array is treated as one instance.
    theta : array-like, shape (n,)
        Coefficients.  Include the bias in both ``X`` and ``theta``, or
        pass ``add_bias=True`` to have a ones column prepended.
    add_bias : bool
        Prepend a ones column to ``X`` before the product.

    Returns
    -------
    result : RichResult
        Keys: p_hat, logits, prediction, estimate, n, method.

    Examples
    --------
    ``sigma(0) = 0.5`` exactly, and the boundary is where the logit is 0:

    >>> r = geron_logistic_probability([[1.0, 0.0]], [0.0, 3.0])
    >>> float(r["p_hat"][0])
    0.5

    ``sigma(2) = 1/(1+e^-2)``:

    >>> round(float(geron_logistic_probability([[1.0]], [2.0])["p_hat"][0]), 9)
    0.880797078

    Symmetry ``sigma(-z) = 1 - sigma(z)`` holds exactly at the
    floating-point level for moderate z:

    >>> a = float(geron_logistic_probability([[1.0]], [1.5])["p_hat"][0])
    >>> b = float(geron_logistic_probability([[-1.0]], [1.5])["p_hat"][0])
    >>> bool(abs(a + b - 1.0) < 1e-15)
    True

    Extreme logits saturate instead of overflowing:

    >>> float(geron_logistic_probability([[1.0]], [-800.0])["p_hat"][0])
    0.0

    References
    ----------
    Géron Ch 4
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.ndim != 2:
        raise ValueError(f"geron_logistic_probability: X must be 1-D or 2-D, got ndim={A.ndim}")
    if A.size == 0:
        raise ValueError("geron_logistic_probability: X is empty")
    if add_bias:
        A = add_bias_column(A)
    t = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()
    if t.size != A.shape[1]:
        raise ValueError(
            f"geron_logistic_probability: theta has {t.size} coefficients but X has {A.shape[1]} columns"
            + (" (after adding the bias column)" if add_bias else "")
        )
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(t)):
        raise ValueError("geron_logistic_probability: X and theta must be finite")

    logits = A @ t
    p = sigmoid(logits)

    return RichResult(
        title="Logistic probability",
        summary_lines=[
            ("Instances", int(A.shape[0])),
            ("Mean p_hat", float(np.mean(p))),
            ("Predicted positives", int(np.count_nonzero(p >= 0.5))),
        ],
        interpretation="The decision boundary is the linear set theta^T x = 0, where p_hat = 0.5.",
        payload={
            "p_hat": p,
            "logits": logits,
            "prediction": (p >= 0.5).astype(int),
            "estimate": float(np.mean(p)),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlogp: logistic p_hat = sigma(theta^T x) with an overflow-safe sigmoid"


# compact alias per ledger/NAMING.md
addbiascolumn = add_bias_column
