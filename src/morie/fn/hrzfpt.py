# morie.fn -- function file (rootcoder007/morie)
"""First-passage time estimation in panel data model."""

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_first_passage_time"]


def horowitz_first_passage_time(theta, y1, y_star, x, beta, f_U, grid_u,
                                f_eps, grid_z):
    r"""Probability that a first-passage time exceeds theta in a panel
    model (Horowitz Sec. 5.2.3), equation (5.20):

    .. math:: P(\theta \mid y_1, y^*, x) =
              \frac{1}{f_W(y_1 - \beta'x_1)}
              \int_{-\infty}^{\infty} f_\varepsilon(y_1 - \beta'x_1 - u)
              \left[\prod_{k=2}^{\theta}
              F_\varepsilon(y^* - \beta'x_k - u)\right] f_U(u)\,du.

    The first-passage time is the first period in which
    :math:`Y_{jt}` exceeds a threshold :math:`y^*`, so the event
    "it has not happened by theta" is the JOINT event
    :math:`Y_{j2} \le y^*, \dots, Y_{j\theta} \le y^*` given the
    initial value and the covariates. This is the reason
    Sec. 5.2 estimates :math:`f_U` and :math:`f_\varepsilon` at all:
    they are rarely interesting in themselves, but the first-passage
    distribution cannot be written without BOTH of them.

    The integral over u is what handles the individual effect
    correctly. Conditional on :math:`U_j = u` the periods are
    independent, so their probabilities multiply -- the product in the
    integrand -- but unconditionally they are DEPENDENT, because they
    share :math:`U_j`. Integrating the product against :math:`f_U`
    rather than multiplying unconditional probabilities is the whole
    difference, and it is why a naive
    :math:`\prod_k P(Y_k \le y^*)` is wrong.

    :math:`F_\varepsilon` is obtained by integrating ``f_eps`` over
    the supplied grid, and :math:`f_W` is the convolution of ``f_U``
    and ``f_eps`` evaluated at the initial residual, so no extra
    inputs are needed beyond the two densities.

    Parameters
    ----------
    theta : int
        Horizon, at least 2.
    y1 : float
        Initial value :math:`Y_{j1}`.
    y_star : float
        Threshold.
    x : array-like, shape (theta, d)
        Covariates for periods 1..theta.
    beta : array-like, shape (d,)
        Coefficients.
    f_U, grid_u : array-like
        The density of U on its grid.
    f_eps, grid_z : array-like
        The density of eps on its grid.

    Returns
    -------
    RichResult
        keys: ``probability``, ``theta``, ``f_W_at_initial``,
        ``periods_conditionally_independent`` (True),
        ``periods_marginally_independent`` (False), ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 5.2.3, eq. (5.20).
    """
    th = int(theta)
    if th < 2:
        raise ValueError(f"theta must be at least 2, got {th}.")
    b = np.asarray(beta, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[1] != b.size:
        X = X.T
    if X.shape[1] != b.size:
        raise ValueError(f"x must have {b.size} columns to match beta.")
    if X.shape[0] < th:
        raise ValueError(
            f"x has {X.shape[0]} periods but theta = {th} needs {th}.")
    gu = np.asarray(grid_u, dtype=float).ravel()
    fu = np.asarray(f_U, dtype=float).ravel()
    gz = np.asarray(grid_z, dtype=float).ravel()
    fe = np.asarray(f_eps, dtype=float).ravel()
    if fu.size != gu.size:
        raise ValueError(f"f_U has {fu.size} entries for {gu.size} grid points.")
    if fe.size != gz.size:
        raise ValueError(
            f"f_eps has {fe.size} entries for {gz.size} grid points.")
    if np.any(fu < 0) or np.any(fe < 0):
        raise ValueError("densities must be non-negative.")

    # F_eps by integrating f_eps, clipped to [0, 1] since the
    # deconvolved density is only approximately a density
    F_cum = np.concatenate([[0.0], np.cumsum(
        np.diff(gz) * (fe[:-1] + fe[1:]) / 2.0)])
    total = F_cum[-1]
    F_cum = F_cum / total if total > 0 else F_cum

    def F_eps(v):
        return np.clip(np.interp(v, gz, F_cum), 0.0, 1.0)

    def f_eps_at(v):
        return np.clip(np.interp(v, gz, fe, left=0.0, right=0.0), 0.0, None)

    idx1 = float(y1 - X[0] @ b)
    # conditional on U = u the periods are independent, so multiply;
    # unconditionally they are not, so the product is integrated
    # against f_U rather than the factors being multiplied outright
    prod = np.ones(gu.size)
    for k in range(1, th):
        prod *= F_eps(y_star - float(X[k] @ b) - gu)
    integrand = f_eps_at(idx1 - gu) * prod * fu
    numer = float(np.trapezoid(integrand, gu))
    # f_W = f_U * f_eps evaluated at the initial residual
    f_W = float(np.trapezoid(f_eps_at(idx1 - gu) * fu, gu))
    if f_W <= 0:
        raise ValueError(
            "f_W vanishes at the initial residual; the conditioning event "
            "has zero estimated density there.")
    return RichResult(payload={
        "probability": float(np.clip(numer / f_W, 0.0, 1.0)),
        "theta": th, "f_W_at_initial": f_W,
        "periods_conditionally_independent": True,
        "periods_marginally_independent": False,
        "method": "(5.20): the product is integrated against f_U, since periods share U_j"})


def cheatsheet():
    return "hrzfpt: periods are independent GIVEN U_j only -- integrate the product, don't multiply margins"
