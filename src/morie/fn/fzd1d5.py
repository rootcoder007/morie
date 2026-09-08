# morie.fn -- function file (rootcoder007/morie)
"""Check conditions D1-D5 of the boundary-free kernel estimators."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfassum", "fauzi_conditions_d1_d5"]


def bfassum(kernel=None, g=None, h=None, n=None, smooth=None, tol=1e-6, lo=-8.0, hi=8.0, ngrid=4001):
    r"""Check conditions D1-D5 of the boundary-free kernel estimators.

    The five conditions of Sec. 5.2, checked rather than asserted:

    D1. ``K`` is non-negative, continuous, symmetric at 0.
    D2. :math:`\int v^2K(v)dv` is finite and :math:`\int K(v)dv = 1`.
    D3. :math:`h>0` with :math:`h\to0`, :math:`nh\to\infty`; checked in
        its finite-sample form ``0 < h < 1`` and ``n*h > 1``.
    D4. ``g`` is an INCREASING bijection from the real line onto the
        support :math:`\Omega`; checked by monotonicity on a fixed grid.
    D5. :math:`f_X` and ``g`` are twice differentiable -- a smoothness
        claim, reported as ``None`` unless the caller asserts it.

    The book says something worth repeating: it is SUFFICIENT for ``g`` to
    be bijective, and the increasing property in D4 is imposed only to
    make the proofs simpler. So a decreasing bijection is not wrong, it is
    merely outside what is proved here -- which is why a decreasing ``g``
    returns ``d4 = False`` with the direction reported in ``monotone``,
    rather than raising.

    D2 is checked as a pair. A kernel can integrate to 1 and still have an
    infinite second moment (Cauchy), so mass and :math:`\mu_2` are
    reported separately.

    Parameters
    ----------
    kernel : callable, optional
        ``K(v)``; defaults to the Gaussian density.
    g : callable, optional
        The transformation, for D4.
    h : float, optional
        Bandwidth, for D3.
    n : int, optional
        Sample size, for D3.
    smooth : bool, optional
        The caller's assertion of D5.
    tol : float, default 1e-6
        Tolerance for the D1/D2 checks.
    lo, hi, ngrid : float, float, int
        Fixed quadrature window and node count.

    Returns
    -------
    RichResult
        Keys ``d1``, ``d2``, ``d3``, ``d4``, ``d5``, ``mass``, ``mu2``, ``monotone``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), conditions D1-D5 of Sec. 5.2.
    """
    from . import _stats_core as stats

    if kernel is None:
        kfun = lambda t: float(stats.norm.pdf(t))
    elif callable(kernel):
        kfun = lambda t: float(kernel(t))
    else:
        raise ValueError("kernel must be None or a callable K(v).")
    v = np.linspace(float(lo), float(hi), int(ngrid))
    kv = np.asarray([kfun(float(t)) for t in v], dtype=float)
    mass = float(np.trapezoid(kv, v))
    sym = float(np.max(np.abs(kv - kv[::-1])))
    mu2 = float(np.trapezoid(v ** 2 * kv, v))
    d1 = bool(np.all(kv >= 0) and sym < float(tol))
    d2 = bool(np.isfinite(mu2) and abs(mass - 1.0) < float(tol))
    if h is None or n is None:
        d3 = None
    else:
        d3 = bool(float(h) > 0 and float(h) < 1 and int(n) * float(h) > 1)
    if g is None:
        d4 = None
        monotone = "unknown"
    else:
        gv = np.asarray([float(g(float(t))) for t in v], dtype=float)
        dv = np.diff(gv)
        if np.all(dv > 0):
            monotone = "increasing"
        elif np.all(dv < 0):
            monotone = "decreasing"
        else:
            monotone = "neither"
        d4 = bool(monotone == "increasing")
    d5 = None if smooth is None else bool(smooth)
    return RichResult(
        payload={
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "d4": d4,
            "d5": d5,
            "mass": mass,
            "mu2": mu2,
            "monotone": monotone,
            "method": "conditions D1-D5 of the boundary-free kernel estimators",
        }
    )


fauzi_conditions_d1_d5 = bfassum


def cheatsheet():
    return "fzd1d5: D1-D5 checked; D4's increasing requirement is for proof convenience, not necessity"


# CANONICAL TEST
# >>> import math
# >>> r = bfassum(h=0.1, n=100, g=math.exp)
# >>> r['d1'] and r['d2'] and r['d3'] and r['d4']
# True
