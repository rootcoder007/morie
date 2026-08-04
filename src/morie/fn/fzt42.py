# morie.fn -- function file (rootcoder007/morie)
"""Bias and variance of the second cumulative-survival estimator (Theorem 4.2)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["srvbv2", "fauzi_thm4_2_surv2_bias_var"]


def srvbv2(t, n, h, surv, cumsurv, dg, d2g, density, mu2=1.0):
    r"""Bias and variance of the second cumulative-survival estimator (Theorem 4.2).

    Theorem 4.2, Eqs. (4.19)-(4.22):

    .. math::
        \mathrm{Bias}[S_{X,2}(t)] &= \tfrac{h^2}2 b_3(t)\mu_2(K) + o(h^2),\\
        \mathrm{Var}[S_{X,2}(t)] &= \tfrac1n[2\bar S_X(t) - S_X^2(t)]
            + o(\tfrac hn),

    with
    :math:`b_3(t) = [g'(g^{-1}(t))]^2f_X(t) - g''(g^{-1}(t))S_X(t)`
    from (4.21).

    The variance is IDENTICAL to that of :math:`S_{X,1}` in Theorem 4.1 --
    same expression, not merely the same order. Only the bias
    coefficients differ, :math:`b_2` versus :math:`b_3`. Sec. 4.2 draws
    the practical conclusion: the two are statistically equivalent, and
    :math:`m_{X,1}` is preferred only because it preserves the analytic
    relationship between the survival and cumulative-survival estimates.

    Compare :math:`b_2` (4.15), which carries an integral of
    :math:`g'g''f_X\circ g` over :math:`[g^{-1}(t),\infty)`, against
    :math:`b_3`, which is purely local. That is why :math:`b_3` is cheap
    and :math:`b_2` is not.

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
    cumsurv : float
        ``bar S_X(t)``.
    dg, d2g : float
        ``g'(g^{-1}(t))`` and ``g''(g^{-1}(t))``.
    density : float
        ``f_X(t)``.
    mu2 : float, default 1.0
        ``int y^2 K(y) dy``.

    Returns
    -------
    RichResult
        Keys ``bias``, ``variance``, ``b3``, ``cov``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 4.2, Eqs. (4.19)-(4.22).
    """
    n = int(n)
    h = float(h)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    b3 = float(dg) ** 2 * float(density) - float(d2g) * float(surv)
    bias = (h * h / 2.0) * b3 * float(mu2)
    var = (2.0 * float(cumsurv) - float(surv) ** 2) / n
    cov = float(surv) * (1.0 - float(surv)) / n
    return RichResult(
        payload={
            "bias": float(bias),
            "variance": float(var),
            "b3": float(b3),
            "cov": float(cov),
            "h": h,
            "n": n,
            "method": "second cumulative-survival estimator moments (Theorem 4.2)",
        }
    )


fauzi_thm4_2_surv2_bias_var = srvbv2


def cheatsheet():
    return "fzt42: Thm 4.2: same variance as S_X,1 exactly; only b_3 differs, and it is local (4.21)"


# CANONICAL TEST
# >>> r = srvbv2(t=1.0, n=100, h=0.1, surv=0.4, cumsurv=0.5, dg=1.0, d2g=0.0, density=0.35)
# >>> abs(r['b3'] - 0.35) < 1e-15
# True
