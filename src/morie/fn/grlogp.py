# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Predicted probability for binary logistic regression."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_logistic_regression_probability"]

_METHOD = "Logistic regression probability (Eq 4-15)"


def _sigmoid(z):
    """Logistic sigmoid, branch-split so neither exp overflows."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def geron_logistic_regression_probability(X, theta):
    r"""Géron Eq 4-15.

    .. math::
        \hat p = \sigma(\theta^{\mathsf T} x),\qquad
        \sigma(t) = \frac{1}{1 + e^{-t}}

    The sigmoid is evaluated by branch: ``1/(1+e^{-z})`` for
    :math:`z \ge 0` and ``e^z/(1+e^z)`` below.  Each branch exponentiates
    a non-positive number, so nothing overflows and a logit of ``-800``
    returns ``0.0`` instead of ``nan``.

    ``grlogc`` (log loss) and ``grlogg`` (its gradient) both get their
    probabilities from here.

    Parameters
    ----------
    X : array-like, shape (n,) or (m, n)
    theta : array-like, shape (n,)
        Prepend the bias column to ``X`` yourself if the model has one.

    Returns
    -------
    RichResult
        Payload keys ``probability``, ``logit``, ``prediction``
        (thresholded at 0.5), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 4, Eq 4-15 (Logistic Regression probability).

    Examples
    --------
    A zero logit is the coin flip:

    >>> r = geron_logistic_regression_probability([[1.0, 0.0]], [0.0, 1.0])
    >>> r["probability"]
    [0.5]

    Logit 2 gives ``1/(1+e^-2)``:

    >>> r2 = geron_logistic_regression_probability([[1.0, 2.0]], [0.0, 1.0])
    >>> round(r2["probability"][0], 9)
    0.880797078

    Extreme logits saturate but stay finite and in range:

    >>> geron_logistic_regression_probability([[-800.0]], [1.0])["probability"]
    [0.0]
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got ndim {X.ndim}.")
    if X.shape[0] == 0:
        raise ValueError("X has no rows.")
    if X.shape[1] != theta.size:
        raise ValueError(f"X has {X.shape[1]} columns but theta has {theta.size} entries.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(theta)):
        raise ValueError("X and theta must be finite.")

    z = X @ theta
    p = _sigmoid(z)

    return RichResult(
        title="Logistic regression probability",
        summary_lines=[("Instances", int(X.shape[0])), ("Mean p", float(p.mean()))],
        payload={
            "probability": p.tolist(),
            "logit": z.tolist(),
            "prediction": (p >= 0.5).astype(int).tolist(),
            "estimate": p.tolist(),
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlogp: p_hat = sigmoid(X theta), overflow-safe branch form -- Geron Eq 4-15"
