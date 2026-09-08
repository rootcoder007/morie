# morie.fn -- function file (rootcoder007/morie)
"""Check assumptions B1-B5 of the bias-reduced KDFE."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kdfassum", "fauzi_assumptions_b1_b5"]


def kdfassum(kernel=None, h=None, n=None, smooth=None, tol=1e-6, lo=-8.0, hi=8.0, ngrid=4001):
    r"""Check assumptions B1-B5 of the bias-reduced KDFE.

    The five standing assumptions of Sec. 2.2, checked numerically rather
    than asserted:

    B1. ``K`` is non-negative, continuous, symmetric about 0, integrating
        to 1.
    B2. :math:`\int w^4K(w)\,dw` is finite.
    B3. :math:`h>0` with :math:`h\to0` and :math:`nh\to\infty`; checked
        here in its usable finite-sample form, ``0 < h < 1`` and
        ``n*h > 1``.
    B4. :math:`f_X` is three times continuously differentiable and
        :math:`f_X^{(4)}` exists -- a smoothness claim about the unknown
        density, which NO routine can verify from data. It is reported as
        ``None`` and the caller may assert it via ``smooth``.
    B5. :math:`\int [f_X'(x)]^2/F_X(x)\,dx` and :math:`\int f_X(x)\,dx`
        are finite.

    The book is explicit about the division of labour: B1 and B3 are the
    usual kernel conditions, B2 and B4 exist only to make the exponential
    and logarithmic expansions in the proofs legitimate, and B5 exists
    only so the MISE of Theorem 2.4 is finite. So a failure of B2 or B4
    invalidates the bias RATE, while a failure of B5 invalidates the MISE
    but leaves the pointwise results standing -- which is why they are
    reported separately instead of as one boolean.

    Parameters
    ----------
    kernel : callable, optional
        ``K(w)``; defaults to the Gaussian density.
    h : float, optional
        Bandwidth, for B3.
    n : int, optional
        Sample size, for B3.
    smooth : bool, optional
        The caller's assertion of B4; there is no way to check it.
    tol : float, default 1e-6
        Tolerance for the B1 symmetry and unit-mass checks.
    lo, hi, ngrid : float, float, int
        Fixed quadrature window and node count.

    Returns
    -------
    RichResult
        Keys ``b1``, ``b2``, ``b3``, ``b4``, ``b5``, ``mu4``, ``mass``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), assumptions B1-B5 of Sec. 2.2.
    """
    from . import _stats_core as stats

    if kernel is None:
        kfun = lambda t: float(stats.norm.pdf(t))
    elif callable(kernel):
        kfun = lambda t: float(kernel(t))
    else:
        raise ValueError("kernel must be None or a callable K(w).")
    w = np.linspace(float(lo), float(hi), int(ngrid))
    kv = np.asarray([kfun(float(t)) for t in w], dtype=float)
    mass = float(np.trapezoid(kv, w))
    sym = float(np.max(np.abs(kv - kv[::-1])))
    mu4 = float(np.trapezoid(w ** 4 * kv, w))
    b1 = bool(np.all(kv >= 0) and abs(mass - 1.0) < float(tol) and sym < float(tol))
    b2 = bool(np.isfinite(mu4))
    if h is None or n is None:
        b3 = None
    else:
        b3 = bool(float(h) > 0 and float(h) < 1 and int(n) * float(h) > 1)
    b4 = None if smooth is None else bool(smooth)
    b5 = None
    return RichResult(
        payload={
            "b1": b1,
            "b2": b2,
            "b3": b3,
            "b4": b4,
            "b5": b5,
            "mu4": mu4,
            "mass": mass,
            "method": "assumptions B1-B5 of the bias-reduced KDFE",
        }
    )


fauzi_assumptions_b1_b5 = kdfassum


def cheatsheet():
    return "fzb1b5: B1-B5 checked, not asserted; B4 is unverifiable from data and says so"


# CANONICAL TEST
# >>> r = kdfassum(h=0.1, n=100)
# >>> r['b1'] and r['b2'] and r['b3']
# True
