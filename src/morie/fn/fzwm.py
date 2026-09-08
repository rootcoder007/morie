# morie.fn -- function file (rootcoder007/morie)
"""Mean and variance of the smoothed Wilcoxon signed rank statistic."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["swilmom", "fauzi_wilcoxon_moments"]


def swilmom(n, gtheta=0.5, projint=1.0 / 3.0):
    r"""Mean and variance of the smoothed Wilcoxon signed rank statistic.

    Sec. 5.3.1. With
    :math:`\tilde W = \tfrac{n(n+1)}2 - \sum_{i\le j}K(-(X_i+X_j)/(2h_n))`,

    .. math::
        E_\theta(\tilde W) &= \frac{n(n+1)}2\{G(\theta) + O(h_n^2)\}, \\
        V_\theta(\tilde W) &= n(n+1)^2\Big\{\!\int\! F^2(u+2\theta)f(u)du
            - G^2(\theta) + O(h_n^2)\Big\},

    with :math:`G` the half-sum distribution function.

    :math:`\tilde W` is a U-statistic, which is why its variance has the
    :math:`n(n+1)^2` shape rather than the sign test's :math:`n`: the
    leading term is :math:`n` times the variance of the FIRST PROJECTION
    :math:`\int F^2(u+2\theta)f(u)du - G^2(\theta)`, and each of the
    :math:`\binom{n+1}2` pairs contributes through it.

    Under :math:`H_0` with symmetric :math:`F`, :math:`G(0)=1/2` and the
    projection integral is :math:`1/3`, giving the familiar
    :math:`n(n+1)^2/12`. Those null values are the defaults.

    Unlike the sign test, :math:`\tilde W` is NOT distribution-free -- the
    book says so plainly -- but its asymptotic moments under :math:`H_0`
    do not depend on :math:`F`, which is all Theorem 5.9 needs.

    This module previously carried a copy of a Kolmogorov-Smirnov
    implementation. It now computes the moments.

    Verified against the primary source: Maesono, Y., Moriyama, T. and
    Lu, M. (2018), *AISM* 70(5):969-982 (arXiv:1610.02145), Sec. 3.

    Parameters
    ----------
    n : int
        Sample size.
    gtheta : float, default 0.5
        ``G(theta)``; 1/2 under the null.
    projint : float, default 1/3
        ``int F^2(u + 2 theta) f(u) du``; 1/3 under the null.

    Returns
    -------
    RichResult
        Keys ``mean``, ``variance``, ``se``, ``projvar``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Sec. 5.3.1; Maesono, Moriyama and Lu (2018), AISM 70:969-982.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    g = float(gtheta)
    if not 0.0 <= g <= 1.0:
        raise ValueError(f"G(theta) must lie in [0, 1], got {g}.")
    projvar = float(projint) - g * g
    mean = n * (n + 1.0) / 2.0 * g
    var = n * (n + 1.0) ** 2 * projvar
    return RichResult(
        payload={
            "mean": float(mean),
            "variance": float(var),
            "se": float(np.sqrt(var)) if var > 0 else float("nan"),
            "projvar": float(projvar),
            "n": n,
            "method": "smoothed Wilcoxon signed rank mean and variance (Sec. 5.3.1)",
        }
    )


fauzi_wilcoxon_moments = swilmom


def cheatsheet():
    return "fzwm: smoothed Wilcoxon moments; U-statistic shape n(n+1)^2; previously a copied KS body"


# CANONICAL TEST
# >>> r = swilmom(n=10)
# >>> abs(r['variance'] - 10 * 121 / 12) < 1e-9
# True
