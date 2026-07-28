# morie.fn -- function file (rootcoder007/morie)
"""Functional delta method."""

import numpy as np

from ._kosorok import hadamard_derivative
from ._richresult import RichResult

__all__ = ["kosorok_ch2_functional_delta_method"]


def kosorok_ch2_functional_delta_method(phi, X_n, theta, r_n, h=None):
    r"""Functional delta method:

    if :math:`r_n(X_n - \theta) \Rightarrow X` and phi is Hadamard
    differentiable at theta, then

    .. math:: r_n\big(\phi(X_n) - \phi(\theta)\big)
              \Rightarrow \phi'_\theta(X).

    Returns both sides at the observed :math:`X_n`: the actual scaled
    increment and the linear approximation :math:`\phi'_\theta` of
    the scaled deviation. Their gap is the delta-method remainder,
    which the theorem says is :math:`o_P(1)` -- reported, not assumed,
    so a phi that is NOT differentiable shows up as a remainder that
    does not shrink.

    Parameters
    ----------
    phi : callable
        The functional.
    X_n : array-like or float
        The statistic.
    theta : array-like or float
        The centring value.
    r_n : float
        Scaling rate (typically sqrt(n)).
    h : array-like, optional
        Direction for the derivative; the observed deviation
        ``X_n - theta`` if omitted.

    Returns
    -------
    RichResult
        keys: ``scaled_increment``, ``linear_approximation``,
        ``remainder``, ``derivative``, ``derivative_converged``,
        ``r_n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (the functional delta method).
    """
    r_n = float(r_n)
    if r_n <= 0:
        raise ValueError(f"r_n must be positive, got {r_n}.")
    Xn = np.asarray(X_n, dtype=float)
    th = np.asarray(theta, dtype=float)
    if Xn.shape != th.shape:
        raise ValueError("X_n and theta must have the same shape.")
    dev = Xn - th
    direction = dev if h is None else np.asarray(h, dtype=float)
    der, _drift, ok = hadamard_derivative(phi, th, direction)
    actual = r_n * (np.asarray(phi(Xn), dtype=float)
                    - np.asarray(phi(th), dtype=float))
    linear = r_n * np.asarray(der, dtype=float) if h is None else np.asarray(
        der, dtype=float
    )
    return RichResult(
        payload={"scaled_increment": actual, "linear_approximation": linear,
                 "remainder": actual - linear, "derivative": der,
                 "derivative_converged": ok, "r_n": r_n,
                 "method": "r_n(phi(X_n) - phi(theta)) vs phi'_theta; remainder shown"}
    )


def cheatsheet():
    return "ksr042: remainder is reported, so non-differentiable phi is visible"
