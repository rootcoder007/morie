# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient of logistic regression cost."""

import numpy as np

from ._richresult import RichResult
from .hmlogp import geron_logistic_probability

__all__ = ["geron_logistic_gradient"]

_METHOD = "Logistic regression cost gradient"


def geron_logistic_gradient(X, y, theta, add_bias=False):
    """
    Gradient of logistic regression cost.

    Formula: grad J = (1/m) X^T (sigmoid(X theta) - y)

    Identical in form to the linear-regression gradient with ``p_hat``
    in place of ``X theta`` -- the sigmoid and the log loss are chosen
    together precisely so that everything else cancels.  The
    probabilities are delegated to
    :func:`morie.fn.hmlogp.geron_logistic_probability`.

    The Hessian ``(1/m) X^T diag(p(1-p)) X`` is returned too: it is
    positive semi-definite for every ``theta``, which is the proof that
    the cost is convex and has no local minima.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix.
    y : array-like, shape (m,)
        Labels in {0, 1}.
    theta : array-like, shape (n,)
        Coefficients.
    add_bias : bool
        Prepend a ones column to ``X``.

    Returns
    -------
    result : RichResult
        Keys: gradient, hessian, p_hat, residuals, estimate, n, method.

    Examples
    --------
    At ``theta = 0`` every probability is 0.5, so the gradient is the
    mean of ``x_i * (0.5 - y_i)``.  With x = 1 and labels 1, 0 those
    cancel exactly:

    >>> r = geron_logistic_gradient([[1.0], [1.0]], [1, 0], [0.0])
    >>> float(r["gradient"][0])
    0.0

    With both labels 1 the gradient is ``(0.5 - 1) = -0.5``:

    >>> float(geron_logistic_gradient([[1.0], [1.0]], [1, 1], [0.0])["gradient"][0])
    -0.5

    Checked against a central difference of the log loss:

    >>> from morie.fn.hmlogcl import geron_logistic_cost
    >>> X = [[1.0, 0.5], [1.0, -1.5], [1.0, 2.0]]
    >>> y = [1, 0, 1]
    >>> th = [0.3, -0.7]
    >>> h = 1e-6
    >>> up = geron_logistic_cost(X, y, [0.3, -0.7 + h])["cost"]
    >>> dn = geron_logistic_cost(X, y, [0.3, -0.7 - h])["cost"]
    >>> g = float(geron_logistic_gradient(X, y, th)["gradient"][1])
    >>> bool(abs((up - dn) / (2 * h) - g) < 1e-7)
    True

    The Hessian is positive semi-definite:

    >>> H = geron_logistic_gradient(X, y, th)["hessian"]
    >>> bool(np.all(np.linalg.eigvalsh(H) >= -1e-12))
    True

    References
    ----------
    Géron Ch 4
    """
    yy = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if yy.size == 0:
        raise ValueError("geron_logistic_gradient: y is empty")
    if not np.all(np.isin(yy, (0.0, 1.0))):
        raise ValueError(
            f"geron_logistic_gradient: y must contain only 0 and 1, got distinct values {np.unique(yy).tolist()}"
        )

    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    inner = geron_logistic_probability(A, theta, add_bias=add_bias)
    p = inner["p_hat"]
    if add_bias:
        A = np.hstack([np.ones((A.shape[0], 1)), A])
    if p.size != yy.size:
        raise ValueError(f"geron_logistic_gradient: X has {p.size} rows but y has {yy.size} entries")

    m = yy.size
    resid = p - yy
    grad = (A.T @ resid) / m
    w = p * (1.0 - p)
    hess = (A.T * w) @ A / m

    return RichResult(
        title="Logistic gradient",
        summary_lines=[("Parameters", int(grad.size)), ("||grad||", float(np.linalg.norm(grad)))],
        interpretation=(
            "Same shape as the linear-regression gradient with p_hat replacing the linear output; "
            "the Hessian is PSD, so the cost is convex."
        ),
        payload={
            "gradient": grad,
            "hessian": hess,
            "p_hat": p,
            "residuals": resid,
            "estimate": float(np.linalg.norm(grad)),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlogg: logistic gradient (1/m) X^T (p_hat - y), plus the PSD Hessian"
