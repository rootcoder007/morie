# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full gradient vector of the linear-regression MSE cost."""

import numpy as np

from ._richresult import RichResult
from .grmse import geron_linreg_mse_cost

__all__ = ["geron_ch4_mse_gradient_vector"]

_METHOD = "MSE gradient vector (Eq 4-7)"


def geron_ch4_mse_gradient_vector(X, y, theta):
    r"""Géron Eq 4-7, the whole gradient in one matrix product.

    .. math::
        \nabla_\theta \mathrm{MSE}(\theta)
        = \frac{2}{m} X^{\mathsf T}\,(X\theta - y)

    The ``2/m`` is not cosmetic: it is what makes the gradient
    scale-free in the number of instances, so a learning rate tuned on
    1 000 rows still works on 100 000.

    The cost itself comes from
    :func:`morie.fn.grmse.geron_linreg_mse_cost`; this module adds only
    the derivative.  Mini-batch and early-stopping drivers
    (``grmgd``, ``greast``) call this function per step.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    theta : array-like, shape (n,)

    Returns
    -------
    RichResult
        Payload keys ``gradient``, ``cost``, ``grad_norm``,
        ``estimate`` (= ``gradient``), ``n``, ``method``.

    References
    ----------
    Geron (2026), Ch 4, Eq 4-7, p. 146.

    Examples
    --------
    Two rows, zero parameters: residual is ``-y``, so the gradient is
    ``(2/2) X^T (-y) = [-3, -5]``:

    >>> X = [[1.0, 1.0], [1.0, 2.0]]
    >>> r = geron_ch4_mse_gradient_vector(X, [1.0, 2.0], [0.0, 0.0])
    >>> r["gradient"]
    [-3.0, -5.0]

    At the least-squares optimum the gradient vanishes:

    >>> g = geron_ch4_mse_gradient_vector(X, [1.0, 2.0], [0.0, 1.0])["gradient"]
    >>> [round(v, 12) for v in g]
    [0.0, 0.0]
    """
    fit = geron_linreg_mse_cost(X, y, theta)          # validates shapes
    X = np.atleast_2d(np.asarray(X, dtype=float))
    resid = np.asarray(fit["residuals"], dtype=float)
    m = X.shape[0]

    grad = (2.0 / m) * (X.T @ resid)

    return RichResult(
        title="MSE gradient vector",
        summary_lines=[("||grad||", float(np.linalg.norm(grad))), ("MSE", fit["cost"])],
        payload={
            "gradient": grad.tolist(),
            "cost": fit["cost"],
            "grad_norm": float(np.linalg.norm(grad)),
            "estimate": grad.tolist(),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grn007: grad MSE = (2/m) X^T (X theta - y) -- Geron Eq 4-7"
