"""Bayesian kriging with a normal-inverse-gamma prior (Kitanidis 1986; Le and Zidek 1992)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bayesian_kriging_nig"]


def bayesian_kriging_nig(X, z, X0, V, V0z, V00, a=0.0, d=0.0, m=None, Qinv=None):
    r"""Predictive distribution of :math:`Z(s_0)` under a conjugate NIG prior.

    Schabenberger & Gotway (2005, Sec. 6.4.3.1, eqs 6.94-6.98): with
    :math:`[Z(s)\mid\beta, \sigma^2] \sim G(X\beta, \sigma^2 V)`, V known, and
    :math:`(\beta, \sigma^2) \sim NIG(a, d, m, Q)`, the posterior has
    :math:`Q^* = (Q^{-1} + X'V^{-1}X)^{-1}`,
    :math:`m^* = Q^*(Q^{-1}m + X'V^{-1}Z)` and
    :math:`a^* = a + m'Q^{-1}m + Z'V^{-1}Z - m^{*\prime}Q^{*-1}m^*` (p. 392),
    and :math:`Z(s_0)\mid Z(s)` is multivariate t with

    .. math::

        E = X_0 m^* + V_{0z}V^{-1}_{zz}(Z - Xm^*),\qquad
        \mathrm{Var} = \frac{a^*}{\nu - 2}\big\{V_{00} - V_{0z}V^{-1}_{zz}V_{z0}
        + (X_0 - V_{0z}V^{-1}_{zz}X)Q^*(X_0 - V_{0z}V^{-1}_{zz}X)'\big\},

    the mean being (6.97) rearranged. :math:`\nu = n + d` for a proper prior
    on :math:`\beta`; with ``Qinv=None`` (a flat prior, :math:`Q^{-1} = 0`)
    :math:`\nu = n + d - p` and the mean is the universal kriging predictor
    (5.29). The printed :math:`\nu^*` of (6.98) weights
    :math:`\hat\beta_{gls} - m` by :math:`X'V^{-1}X`; the exact scale uses
    :math:`a^*`, which equals :math:`a + (n-p)\hat\sigma^2_{gls} +
    (\hat\beta_{gls} - m)'(Q + (X'V^{-1}X)^{-1})^{-1}(\hat\beta_{gls} - m)`.

    Parameters
    ----------
    X : (n, p) array-like
    z : (n,) array-like
    X0 : (n0, p) array-like
    V : (n, n) array-like
        Correlation (or scaled covariance) of the data.
    V0z : (n0, n) array-like
    V00 : (n0, n0) array-like
    a, d : float
        Inverse-gamma hyperparameters (mean ``a / (d - 2)``).
    m : (p,) array-like, optional
        Prior mean of beta (proper prior).
    Qinv : (p, p) array-like, optional
        Prior precision of beta up to sigma^2; ``None`` is flat.

    Returns
    -------
    RichResult
        ``mean``, ``variance`` (matrix), ``scale`` (matrix), ``df``,
        ``a_star``, ``m_star``, ``Q_star``.

    References
    ----------
    Kitanidis, P. K. (1986). Parameter uncertainty in estimation of spatial
    functions: Bayesian analysis. Water Resources Research 22, 499-507.
    Le, N. D. & Zidek, J. V. (1992). Interpolation with uncertain spatial
    covariances: a Bayesian alternative to kriging. Journal of Multivariate
    Analysis 43, 351-374. Schabenberger, O. & Gotway, C. A. (2005).
    Statistical Methods for Spatial Data Analysis. Chapman & Hall/CRC, eqs
    (6.94)-(6.98), pp. 391-394.
    """
    X = np.asarray(X, dtype=float)
    z = np.asarray(z, dtype=float).ravel()
    X0 = np.atleast_2d(np.asarray(X0, dtype=float))
    V = np.asarray(V, dtype=float)
    V0z = np.atleast_2d(np.asarray(V0z, dtype=float))
    V00 = np.atleast_2d(np.asarray(V00, dtype=float))
    n, p = X.shape
    flat = Qinv is None
    Qi = np.zeros((p, p)) if flat else np.asarray(Qinv, dtype=float)
    mm = np.zeros(p) if m is None else np.asarray(m, dtype=float).ravel()
    Vi = np.linalg.inv(V)
    A = X.T @ Vi @ X
    Qs = np.linalg.inv(Qi + A)
    ms = Qs @ (Qi @ mm + X.T @ (Vi @ z))
    astar = float(a + mm @ (Qi @ mm) + z @ (Vi @ z) - ms @ ((Qi + A) @ ms))
    nu = n + float(d) - (p if flat else 0)
    if nu <= 2:
        raise ValueError("the predictive t needs more than 2 degrees of freedom")
    K = V0z @ Vi
    mean = X0 @ ms + K @ (z - X @ ms)
    Dm = X0 - K @ X
    C = V00 - K @ V0z.T + Dm @ Qs @ Dm.T
    C = 0.5 * (C + C.T)
    return RichResult(
        title="Bayesian kriging, normal-inverse-gamma prior (eqs 6.97-6.98)",
        summary_lines=[("df", nu), ("a*", astar), ("prior", "flat" if flat else "NIG")],
        payload={
            "mean": [float(v) for v in mean],
            "variance": C * (astar / (nu - 2.0)),
            "scale": C * (astar / nu),
            "df": nu,
            "a_star": astar,
            "m_star": [float(v) for v in ms],
            "Q_star": Qs,
        },
    )


def cheatsheet():
    return "bkrnig: Bayesian kriging predictive t under NIG(a, d, m, Q); flat prior = universal kriging mean"
