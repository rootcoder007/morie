# morie.fn -- function file (rootcoder007/morie)
"""Constant Conditional Correlation MGARCH."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .dccmd import _univariate_garch11

__all__ = ["vol_ccc_garch"]


def vol_ccc_garch(R_panel, init=None):
    r"""Bollerslev (1990) Constant Conditional Correlation MGARCH.

    The conditional covariance factorises into time-varying marginal
    volatilities and a *constant* correlation matrix:

    .. math::

        H_t = D_t R D_t, \qquad
        D_t = \mathrm{diag}\!\left(\sqrt{h_{1t}}, \dots, \sqrt{h_{kt}}\right)

    with each :math:`h_{jt}` a univariate GARCH(1,1). Because :math:`R`
    does not move, the Gaussian likelihood separates: the marginals are
    estimated series by series, and the maximum-likelihood :math:`R` is
    then the sample correlation of the standardised residuals
    :math:`z_t = D_t^{-1}\varepsilon_t`. No numerical search is needed
    for the second step -- that closed form is the whole point of the
    constant-correlation restriction, and it is what makes CCC the cheap
    baseline that DCC relaxes (see :func:`morie.fn.voldcc.vol_dcc_garch`).

    Parameters
    ----------
    R_panel : array-like, shape (n, k)
        Return panel, n observations x k assets. A transposed panel is
        detected and corrected. k >= 2 is required -- a "constant
        correlation" between one series and itself carries no
        information.
    init : array-like, optional
        Ignored; retained for signature compatibility with the other
        volatility front-ends. The CCC second step is closed-form, so
        there is no starting value to supply.

    Returns
    -------
    RichResult
        keys: ``R`` (k x k constant correlation), ``sigmas`` (n x k
        conditional standard deviations), ``conditional_variance``
        (n x k), ``ll``, ``n``, ``k``, ``method``.

    References
    ----------
    Bollerslev, T. (1990). Modelling the coherence in short-run nominal
    exchange rates: a multivariate generalized ARCH model. *Review of
    Economics and Statistics*, 72(3), 498-505.
    """
    del init  # closed-form second step; nothing to initialise
    X = np.atleast_2d(np.asarray(R_panel, dtype=float))
    if X.shape[0] < X.shape[1]:
        X = X.T
    n, k = X.shape
    if n < 30 or k < 2:
        raise ValueError(f"Need n>=30, k>=2; got n={n}, k={k}.")

    H = np.empty((n, k))
    Z = np.empty((n, k))
    for j in range(k):
        rj = X[:, j] - X[:, j].mean()
        H[:, j] = _univariate_garch11(rj)
        Z[:, j] = rj / np.sqrt(H[:, j] + 1e-12)

    # ML estimate of R under the constant-correlation restriction is the
    # sample correlation of the standardised residuals.
    R = np.corrcoef(Z, rowvar=False)
    sign, logdet_R = np.linalg.slogdet(R)
    if sign <= 0:
        raise ValueError("Standardised residuals give a singular correlation matrix.")
    Rinv = np.linalg.inv(R)

    # log L = -0.5 sum_t [ k log 2pi + log|H_t| + eps_t' H_t^-1 eps_t ]
    # with log|H_t| = log|R| + sum_j log h_jt and the quadratic form in z.
    quad = np.einsum("tj,jl,tl->t", Z, Rinv, Z)
    ll = -0.5 * float(np.sum(k * np.log(2 * np.pi) + logdet_R + np.sum(np.log(H), axis=1) + quad))

    return RichResult(
        title="CCC-GARCH (Bollerslev 1990)",
        payload={
            "R": R,
            "sigmas": np.sqrt(H),
            "conditional_variance": H,
            "ll": ll,
            "loglik": ll,
            "n": int(n),
            "k": int(k),
            "method": "CCC(1,1) two-step Gaussian MLE (numpy)",
        },
    )


def cheatsheet():
    return "volccc: Constant Conditional Correlation MGARCH"


# compact alias per ledger/NAMING.md
volcccgarch = vol_ccc_garch
