# morie.fn -- function file (rootcoder007/morie)
"""Logistic loss -- Boyd & Vandenberghe Sec. 7.1.1."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_logistic_loss"]


def boyd_logistic_loss(u):
    r"""The logistic loss :math:`\phi(u) = \log(1 + e^{-u})`.

    Smooth everywhere, unlike the hinge, and strictly positive everywhere,
    which is the difference that matters. A correctly classified point far
    from the boundary still contributes a small gradient, so it still
    pulls -- there is no analogue of a support vector, and every
    observation influences the fit.

    Computed as :math:`\log(1 + e^{-u})` via ``logaddexp`` so that large
    negative u does not overflow: the naive form returns ``inf`` at
    :math:`u \approx -750` and silently loses the answer.

    Parameters
    ----------
    u : array-like
        Margins :math:`y_i(w^\top x_i + b)`.

    Returns
    -------
    RichResult
        ``loss``, ``total``, ``mean``, ``gradient``, ``probability``
        (:math:`\sigma(u)`), ``max_gradient``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    >>> import numpy as np
    >>> r = boyd_logistic_loss([0.0])
    >>> round(float(r["loss"][0]), 6)
    0.693147

    Strictly positive even far out on the correct side -- so unlike the
    hinge, every point keeps pulling.

    >>> bool(boyd_logistic_loss([20.0])["loss"][0] > 0)
    True

    No overflow where the naive log(1 + exp(-u)) would return inf.

    >>> bool(np.isfinite(boyd_logistic_loss([-800.0])["loss"][0]))
    True
    >>> round(float(boyd_logistic_loss([-800.0])["loss"][0]), 1)
    800.0

    The gradient is bounded by 1 in magnitude, which is what makes the
    logistic loss robust to a badly misclassified point in a way squared
    error is not.

    >>> bool(boyd_logistic_loss([-1e6])["max_gradient"] <= 1.0)
    True
    """
    uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    # logaddexp(0, -u) is log(1 + exp(-u)) without the overflow.
    loss = np.logaddexp(0.0, -uv)
    prob = 1.0 / (1.0 + np.exp(-np.clip(uv, -500, 500)))
    grad = -(1.0 - prob)
    return RichResult(
        title="Logistic loss",
        summary_lines=[("n", int(uv.size)), ("total", float(loss.sum())),
                       ("mean", float(loss.mean()))],
        payload={
            "loss": loss, "total": float(loss.sum()),
            "mean": float(loss.mean()), "gradient": grad,
            "probability": prob,
            "max_gradient": float(np.max(np.abs(grad))),
            "method": "boyd_logistic_loss",
        },
    )


def cheatsheet():
    return "cvxlgr: strictly positive everywhere, so EVERY point pulls -- no support vectors"
