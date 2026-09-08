# morie.fn -- function file (rootcoder007/morie)
"""Naive kernel-smoothed Cramer-von Mises statistic (Eq. 5.4)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kerncvm", "fauzi_naive_kernel_cvm"]


def kerncvm(x, quantile, h=None, ngrid=2001):
    r"""Naive kernel-smoothed Cramer-von Mises statistic (Eq. 5.4).

    Eq. (5.4):

    .. math:: \widehat{CvM} = n\!\int_{-\infty}^{\infty}
              [\hat F_X(x)-F(x)]^2dF(x),

    with :math:`\hat F_X` the naive kernel distribution function
    estimator.

    Unlike the empirical :math:`CvM_n`, this has no finite-sum closed
    form: :math:`\hat F_X` is smooth, so the integral does not collapse
    onto the order statistics. It is evaluated by substituting
    :math:`u = F(x)` and integrating over :math:`u\in(0,1)` on a fixed
    equally spaced grid -- which turns :math:`dF(x)` into :math:`du` and
    needs no density, only the quantile function.

    This module previously carried a copy of the empirical KS body. It now
    computes a Cramer-von Mises statistic, and a smoothed one.

    Theorem 5.1 gives :math:`|CvM_n - \widehat{CvM}| \to_p 0`, so the
    Cramer-von Mises critical values still apply.

    Parameters
    ----------
    x : array-like
        Sample.
    quantile : callable
        The null quantile function ``F^{-1}(u)`` for ``u`` in ``(0, 1)``.
    h : float, optional
        Bandwidth; defaults to the distribution-function rule.
    ngrid : int, default 2001
        Number of ``u``-nodes; fixed, never adapted.

    Returns
    -------
    RichResult
        Keys ``statistic``, ``p_value``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (5.4), Theorem 5.1.
    """
    from . import _stats_core as stats
    from ._fauzi import kdfe_bandwidth

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}.")
    if not callable(quantile):
        raise ValueError("quantile must be a callable F^-1(u).")
    if h is None:
        h = kdfe_bandwidth(xv)
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    m = int(ngrid)
    u = (np.arange(m) + 0.5) / m
    integrand = np.empty(m)
    for i, uu in enumerate(u):
        t = float(quantile(float(uu)))
        khat = float(np.mean(stats.norm.cdf((t - xv) / h)))
        integrand[i] = (khat - float(uu)) ** 2
    stat = float(n * np.mean(integrand))
    pval = 1.0
    if stat > 0:
        acc = 0.0
        for k in range(100):
            acc += float(np.exp(-((4.0 * k + 1.0) ** 2) * np.pi ** 2 / (8.0 * stat)))
        pval = max(0.0, min(1.0, 1.0 - acc * float(np.sqrt(2.0 / stat))))
    return RichResult(
        payload={
            "statistic": stat,
            "p_value": float(pval),
            "h": h,
            "n": int(n),
            "method": "naive kernel-smoothed Cramer-von Mises statistic (Eq. 5.4)",
        }
    )


fauzi_naive_kernel_cvm = kerncvm


def cheatsheet():
    return "fznkc: kernel-smoothed CvM by the u = F(x) substitution; no closed form once F is smoothed"


# CANONICAL TEST
# >>> r = kerncvm([0.1, 0.3, 0.5, 0.7, 0.9], quantile=lambda u: u)
# >>> r['statistic'] > 0
# True
