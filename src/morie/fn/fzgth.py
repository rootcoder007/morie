# morie.fn -- function file (rootcoder007/morie)
"""The function G(theta), distribution function of the half-sum (X_1+X_2)/2."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["hsumcdf", "fauzi_g_theta_distribution"]


def hsumcdf(theta=0.0, cdf=None, density=None, lo=-10.0, hi=10.0, ngrid=4001):
    r"""The function G(theta), distribution function of the half-sum (X_1+X_2)/2.

    Sec. 5.3:

    .. math:: G(\theta) = \int_{-\infty}^{\infty}F(2\theta+u)f(u)\,du,

    the distribution function of :math:`(X_1+X_2)/2`.

    It is the population quantity the Mann-Whitney form of Wilcoxon's
    signed rank statistic estimates: :math:`W = \sum_{i\le j}I(X_i+X_j\ge
    0)` has :math:`E_\theta(\tilde W) = \tfrac{n(n+1)}2\{G(\theta) +
    O(h_n^2)\}`. Under :math:`H_0` with :math:`F` symmetric,
    :math:`G(0) = 1/2`.

    Two equivalent forms appear in the literature,
    :math:`\int F(2\theta+u)f(u)du` and :math:`\int F(2\theta-u)f(u)du`;
    they agree because the integrand is symmetrised by ``f``. The book
    prints the first, and the primary source -- Maesono, Y., Moriyama, T.
    and Lu, M. (2018), "Smoothed nonparametric tests and their
    properties", *Annals of the Institute of Statistical Mathematics*
    70(5):969-982 (arXiv:1610.02145) -- prints both and states they are
    equal. The first is used here.

    Integrated on a fixed trapezoid grid; with ``cdf`` and ``density``
    both defaulting to the standard normal, ``G(0)`` returns 1/2 to
    quadrature accuracy, which is the natural check.

    Parameters
    ----------
    theta : float or array-like, default 0.0
        Location parameter.
    cdf : callable, optional
        ``F``; defaults to the standard normal.
    density : callable, optional
        ``f``; defaults to the standard normal.
    lo, hi : float, default -10.0, 10.0
        Quadrature limits.
    ngrid : int, default 4001
        Number of nodes; fixed, never adapted.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``theta``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Sec. 5.3; Maesono, Moriyama and Lu (2018), AISM 70:969-982.
    """
    from . import _stats_core as stats

    if cdf is None:
        cdf = lambda t: float(stats.norm.cdf(t))
    if density is None:
        density = lambda t: float(stats.norm.pdf(t))
    if not callable(cdf) or not callable(density):
        raise ValueError("cdf and density must be callables.")
    u = np.linspace(float(lo), float(hi), int(ngrid))
    fu = np.asarray([float(density(float(t))) for t in u], dtype=float)
    tv = np.atleast_1d(np.asarray(theta, dtype=float))
    out = np.empty(tv.size)
    for i, th in enumerate(tv):
        fv = np.asarray([float(cdf(2.0 * float(th) + float(t))) for t in u], dtype=float)
        out[i] = float(np.trapezoid(fv * fu, u))
    return RichResult(
        payload={
            "estimate": [float(v) for v in out],
            "theta": [float(v) for v in tv],
            "method": "G(theta), distribution function of (X_1 + X_2)/2",
        }
    )


fauzi_g_theta_distribution = hsumcdf


def cheatsheet():
    return "fzgth: G(theta) = int F(2 theta + u) f(u) du -- what the Mann-Whitney form estimates"


# CANONICAL TEST
# >>> abs(hsumcdf(theta=0.0)['estimate'][0] - 0.5) < 1e-9
# True
