# morie.fn -- function file (rootcoder007/morie)
"""Average derivative estimand."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_average_derivative"]


def hrz_average_derivative(X, y, h=None, weighted=True):
    r"""Density-weighted average derivative (Horowitz Ch. 2):

    .. math:: \delta = E\!\left[f_X(X)\,
              \frac{\partial E(Y|X)}{\partial X}\right]
              = -2\,E\big[f_X'(X)\,Y\big].

    Note this is the DENSITY-WEIGHTED average derivative, not the
    plain :math:`E[\partial E(Y|X)/\partial X]`. The two differ by
    the factor :math:`E[f_X(X)]`: for standard normal X with
    :math:`E(Y|X) = 2X`, the weighted estimand is
    :math:`2\int\phi^2 = 0.564` while the unweighted one is 2. The
    weighting is what the second equality -- and hence the root-n
    property -- depends on, so it is not an incidental choice.

    Integration by parts turns a derivative of an unknown regression
    into an expectation involving the DENSITY derivative, which is why
    this estimand is root-n estimable even though :math:`E(Y|X)`
    itself is not. In an index model :math:`\delta \propto \beta`,
    so it identifies the index direction without optimising over it.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    h : float, optional
        Bandwidth for the density derivative.
    weighted : bool, default True
        Use the density-weighted form.

    Returns
    -------
    RichResult
        keys: ``delta``, ``se``, ``root_n`` (True),
        ``proportional_to_beta`` (True), ``bandwidth``, ``n``, ``d``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (average derivative estimation).
    """
    from ._horowitz import kernel_deriv, silverman_bw

    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError("X must have one row per entry of y.")
    n, d = X.shape
    delta = np.empty(d)
    infl = np.empty((n, d))
    hs = np.empty(d)
    for j in range(d):
        xj = X[:, j]
        hj = silverman_bw(xj) if h is None else float(h)
        hs[j] = hj
        # f-hat'(X_i) by leave-one-out, so an observation does not
        # contribute to its own density derivative
        U = (xj[:, None] - xj[None, :]) / hj
        Kp = kernel_deriv(U)
        np.fill_diagonal(Kp, 0.0)
        # f-hat'(x) = (1/(n h^2)) sum K'((x - X_i)/h). No leading minus:
        # the kernel derivative already carries the sign, and adding
        # another flips the estimand (measured: -0.548 vs +0.548 where
        # theory gives +0.564).
        fprime = Kp.sum(axis=1) / ((n - 1) * hj**2)
        contrib = -2.0 * fprime * y
        delta[j] = float(contrib.mean())
        infl[:, j] = contrib - delta[j]
    se = np.sqrt((infl**2).sum(axis=0)) / n
    return RichResult(payload={"delta": delta if d > 1 else float(delta[0]),
                               "se": se if d > 1 else float(se[0]),
                               "root_n": True, "proportional_to_beta": True,
                               "bandwidth": hs if d > 1 else float(hs[0]),
                               "n": int(n), "d": int(d),
                               "method": "Density-weighted average derivative; root-n by parts"})


def cheatsheet():
    return "hrzade: integration by parts makes it root-n; delta ∝ beta in an index model"
