# morie.fn -- function file (rootcoder007/morie)
"""Bias and variance of the boundary-free survival and cumulative-survival estimators (Theorem 4.1)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["srvbv1", "fauzi_thm4_1_surv_bias_var"]


def srvbv1(t, n, h, surv, cdf, cumsurv, b1, b2, dg, density, mu2=1.0, vw=None):
    r"""Bias and variance of the boundary-free survival and cumulative-survival estimators (Theorem 4.1).

    Theorem 4.1, Eqs. (4.10)-(4.16):

    .. math::
        \mathrm{Bias}[\tilde S_X(t)] &= -\tfrac{h^2}2 b_1(t)\mu_2(K)
            + o(h^2), \\
        \mathrm{Var}[\tilde S_X(t)] &= \tfrac1n \tilde S_X(t)F_X(t)
            - \tfrac hn g'(g^{-1}(t))f_X(t)\!\int\! V(y)W(y)dy
            + o(\tfrac hn), \\
        \mathrm{Bias}[S_{X,1}(t)] &= \tfrac{h^2}2 b_2(t)\mu_2(K)
            + o(h^2), \\
        \mathrm{Var}[S_{X,1}(t)] &= \tfrac1n[2\bar S_X(t) - S_X^2(t)]
            + o(\tfrac hn),

    with
    :math:`b_1(t) = g''(g^{-1}(t))f_X(t) + [g'(g^{-1}(t))]^2f_X'(t)` from
    (4.14) and :math:`b_2` from (4.15).

    Note the SIGNS: the survival estimator's bias carries a minus and the
    cumulative one a plus, because :math:`S = 1 - F` flips the leading
    term while the cumulative survival integrates it back. They are not
    interchangeable and the routine returns both rather than one with a
    flag.

    For the Gaussian kernel :math:`\int V(y)W(y)dy = \int(1-W)W\,dy`,
    which is :math:`1/\sqrt\pi` -- computed in closed form here, since
    :math:`\int\Phi(1-\Phi)` over the line is a standard integral.

    Parameters
    ----------
    t : float
        Evaluation point.
    n : int
        Sample size.
    h : float
        Bandwidth.
    surv : float
        ``S_X(t)``.
    cdf : float
        ``F_X(t)``.
    cumsurv : float
        ``bar S_X(t)``, the cumulative survival.
    b1, b2 : float
        The coefficients (4.14) and (4.15).
    dg : float
        ``g'(g^{-1}(t))``.
    density : float
        ``f_X(t)``.
    mu2 : float, default 1.0
        ``int y^2 K(y) dy``.
    vw : float, optional
        ``int V(y)W(y)dy``; defaults to the Gaussian ``1/sqrt(pi)``.

    Returns
    -------
    RichResult
        Keys ``biassurv``, ``varsurv``, ``biascum``, ``varcum``, ``vw``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 4.1, Eqs. (4.10)-(4.15).
    """
    n = int(n)
    h = float(h)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if vw is None:
        vw = 1.0 / np.sqrt(np.pi)
    vw = float(vw)
    bias_s = -(h * h / 2.0) * float(b1) * float(mu2)
    var_s = float(surv) * float(cdf) / n - h / n * float(dg) * float(density) * vw
    bias_c = (h * h / 2.0) * float(b2) * float(mu2)
    var_c = (2.0 * float(cumsurv) - float(surv) ** 2) / n
    return RichResult(
        payload={
            "biassurv": float(bias_s),
            "varsurv": float(var_s),
            "biascum": float(bias_c),
            "varcum": float(var_c),
            "vw": vw,
            "h": h,
            "n": n,
            "method": "boundary-free survival and cumulative-survival moments (Theorem 4.1)",
        }
    )


fauzi_thm4_1_surv_bias_var = srvbv1


def cheatsheet():
    return "fzt41: Thm 4.1: survival bias carries a MINUS, cumulative survival a PLUS (4.10-4.15)"


# CANONICAL TEST
# >>> r = srvbv1(t=1.0, n=100, h=0.1, surv=0.4, cdf=0.6, cumsurv=0.5,
# ...            b1=0.2, b2=0.3, dg=1.0, density=0.35)
# >>> r['biassurv'] < 0 < r['biascum']
# True
