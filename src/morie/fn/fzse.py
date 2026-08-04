# morie.fn -- function file (rootcoder007/morie)
"""Mean and variance of the smoothed sign test statistic."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["ssgnmom", "fauzi_sign_moments"]


def ssgnmom(n, ftheta=0.5, h=None, f0=None, fpp0=None, a11=None, a13=None):
    r"""Mean and variance of the smoothed sign test statistic.

    Sec. 5.3.1. With
    :math:`\tilde S = n - \sum_iK(-X_i/h_n)`,

    .. math::
        E_\theta(\tilde S) &= n\{F(\theta) + O(h_n^2)\}, \\
        V_\theta(\tilde S) &= n[\{1-F(\theta)\}F(\theta) + O(h_n)].

    Under :math:`H_0` these reduce to :math:`n/2` and :math:`n/4`, and
    Theorem 5.10 refines the variance to
    :math:`n/4 - 2nh_nf(0)A_{1,1} - \tfrac{nh_n^3}3f''(0)A_{1,3} + o(1)`
    with :math:`A_{i,j} = \int K^i(u)k(u)u^j du`. Pass ``a11`` and
    ``a13`` with ``f0``/``fpp0`` to get that refinement; otherwise the
    leading forms are returned and ``refined`` is False.

    The point of the whole construction is in the error terms. The
    ordinary sign test :math:`S` is discrete, so its standardised version
    jumps by :math:`O(n^{-1/2})` and no Edgeworth expansion can be valid
    for it. :math:`\tilde S` is continuous, and under :math:`H_0` its
    leading moments do not depend on :math:`F` at all -- asymptotically
    distribution-free, which is what makes Theorem 5.9 possible.

    This module previously carried a copy of a Kolmogorov-Smirnov
    implementation, returning a KS statistic under the name of the sign
    test's moments. It now computes the moments.

    Verified against the primary source: Maesono, Y., Moriyama, T. and
    Lu, M. (2018), *AISM* 70(5):969-982 (arXiv:1610.02145), Sec. 3.

    Parameters
    ----------
    n : int
        Sample size.
    ftheta : float, default 0.5
        ``F(theta)``; 1/2 under the null.
    h : float, optional
        Bandwidth, needed for the Theorem 5.10 refinement.
    f0, fpp0 : float, optional
        ``f(0)`` and ``f''(0)``.
    a11, a13 : float, optional
        ``A_{1,1}`` and ``A_{1,3}``.

    Returns
    -------
    RichResult
        Keys ``mean``, ``variance``, ``se``, ``refined``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Sec. 5.3.1 and Theorem 5.10; Maesono, Moriyama and Lu (2018), AISM 70:969-982.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    ft = float(ftheta)
    if not 0.0 <= ft <= 1.0:
        raise ValueError(f"F(theta) must lie in [0, 1], got {ft}.")
    mean = float(n) * ft
    var = float(n) * (1.0 - ft) * ft
    refined = False
    if None not in (h, f0, fpp0, a11, a13):
        hh = float(h)
        if hh <= 0:
            raise ValueError(f"bandwidth must be positive, got {hh}.")
        var = (
            n / 4.0
            - 2.0 * n * hh * float(f0) * float(a11)
            - n * hh ** 3 / 3.0 * float(fpp0) * float(a13)
        )
        refined = True
    return RichResult(
        payload={
            "mean": float(mean),
            "variance": float(var),
            "se": float(np.sqrt(var)) if var > 0 else float("nan"),
            "refined": bool(refined),
            "n": n,
            "method": "smoothed sign test mean and variance (Sec. 5.3.1)",
        }
    )


fauzi_sign_moments = ssgnmom


def cheatsheet():
    return "fzse: smoothed sign test moments n F(theta), n F(1-F); previously a copied KS body"


# CANONICAL TEST
# >>> r = ssgnmom(n=100)
# >>> r['mean'] == 50.0 and r['variance'] == 25.0
# True
